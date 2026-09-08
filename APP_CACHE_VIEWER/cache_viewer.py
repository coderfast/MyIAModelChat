"""
Cache Viewer - PyQt5 Graphical Interface for Dataset Cache Inspection

Standalone application to view dataset cache files.
Uses lazy loading with QThread for optimal performance with large datasets.

Usage:
    python cache_viewer.py [path_to_dataset_cache]
    python cache_viewer.py                          # Opens folder selector
"""

import os
import sys
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QLabel, QLineEdit, QComboBox,
    QTextEdit, QSplitter, QHeaderView, QMessageBox, QStatusBar,
    QAbstractItemView, QGroupBox, QFormLayout, QApplication,
    QFileDialog, QPushButton, QProgressBar, QTableView
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex, QTimer
)
from PyQt5.QtGui import QFont


# ==================== CONSTANTS ====================

BLOCK_SIZE = 500


# ==================== CONTINUOUS LOADER WORKER ====================

class ContinuousLoaderWorker(QThread):
    """Worker thread that loads a single block of samples."""
    
    block_ready = pyqtSignal(list, int, int)  # items, offset, total
    error_occurred = pyqtSignal(str)
    
    def __init__(self, indices: List[int], dataset, offset: int, block_size: int, token_lengths: List[int] = None):
        super().__init__()
        self.indices = indices
        self.dataset = dataset
        self.offset = offset
        self.block_size = block_size
        self._is_cancelled = False
        self._token_lengths = token_lengths or []
    
    def cancel(self):
        self._is_cancelled = True
    
    def run(self):
        try:
            total = len(self.indices)
            items = []
            end_idx = min(self.offset + self.block_size, total)
            
            for i in range(self.offset, end_idx):
                if self._is_cancelled:
                    return
                
                idx = self.indices[i]
                row = self.dataset[idx]
                text = row.get('bpe_text', row.get('input_ids', ''))
                source = row.get('source', 'Unknown')
                
                token_count = self._token_lengths[idx] if idx < len(self._token_lengths) else 0
                
                items.append({
                    'index': idx,
                    'text': text,
                    'token_count': token_count,
                    'source': source,
                    'language': row.get('language', '-')
                })
            
            if items and not self._is_cancelled:
                self.block_ready.emit(items, self.offset, total)
        
        except Exception as e:
            if not self._is_cancelled:
                self.error_occurred.emit(str(e))


# ==================== LAZY TABLE MODEL ====================

class LazyTableModel(QAbstractTableModel):
    """Custom table model with lazy loading and source filtering support."""
    
    def __init__(self):
        super().__init__()
        self._data: List[Optional[Dict]] = []
        self._total_count = 0
        self._loaded_count = 0
        self._headers = ["#", "Text", "Token Count", "Source", "Language"]
    
    def rowCount(self, parent=QModelIndex()) -> int:
        return self._total_count
    
    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._headers)
    
    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        
        if role == Qt.DisplayRole:
            row = index.row()
            col = index.column()
            
            if row < len(self._data) and self._data[row] is not None:
                item = self._data[row]
                if col == 0:
                    return str(item['index'])
                elif col == 1:
                    text = item['text']
                    return text[:300] + "..." if len(text) > 300 else text
                elif col == 2:
                    return str(item['token_count'])
                elif col == 3:
                    return item.get('source', 'Unknown')
                elif col == 4:
                    return item.get('language', '-')
            
            # Placeholder for unloaded rows
            if col == 0:
                return str(row)
            elif col == 1:
                return "Loading..."
            elif col == 2:
                return "..."
            elif col == 3:
                return "..."
            elif col == 4:
                return "..."
        
        elif role == Qt.TextAlignmentRole:
            if index.column() in [0, 2]:
                return Qt.AlignRight | Qt.AlignVCenter
        
        return None
    
    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if section < len(self._headers):
                return self._headers[section]
        return None
    
    def set_total_count(self, count: int):
        """Set total number of rows in the dataset."""
        self.beginResetModel()
        self._total_count = count
        self._data = [None] * count
        self._loaded_count = 0
        self.endResetModel()
    
    def add_block(self, items: List[Dict], start_idx: int):
        """Add a block of loaded items by updating existing rows."""
        first_row = start_idx
        last_row = min(start_idx + len(items), self._total_count) - 1
        
        if first_row > last_row:
            return
        
        for i, item in enumerate(items):
            idx = start_idx + i
            if idx < len(self._data):
                self._data[idx] = item
        
        top_left = self.index(first_row, 0)
        bottom_right = self.index(last_row, self.columnCount() - 1)
        self.dataChanged.emit(top_left, bottom_right, [Qt.DisplayRole])
        
        self._loaded_count = sum(1 for x in self._data if x is not None)
    
    def get_item(self, row: int) -> Optional[Dict]:
        """Get item at row if loaded."""
        if 0 <= row < len(self._data):
            return self._data[row]
        return None
    
    def is_loaded(self, row: int) -> bool:
        """Check if a row is loaded."""
        if 0 <= row < len(self._data):
            return self._data[row] is not None
        return False
    
    def get_loaded_count(self) -> int:
        return self._loaded_count
    
    def get_total_count(self) -> int:
        return self._total_count


# ==================== VIRTUAL SCROLL TABLE ====================

class VirtualScrollTable(QTableView):
    """QTableView with virtual scroll - signals when user approaches unloaded data."""
    
    scroll_changed = pyqtSignal(int)  # first_visible_row
    prefetch_needed = pyqtSignal(int)  # first_visible_row
    
    PREFETCH_VIEWPORTS = 20  # prefetch when this many viewports away from loaded boundary
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded_row_count = 0
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
    
    def set_loaded_row_count(self, count: int):
        self._loaded_row_count = count
    
    def _viewport_row_count(self) -> int:
        if not self.model() or self.model().rowCount() == 0:
            return 20
        row_h = self.sizeHintForRow(0)
        if row_h <= 0:
            return 20
        return max(1, self.viewport().height() // row_h)
    
    def _on_scroll(self, value: int):
        viewport_rows = self._viewport_row_count()
        last_visible = value + viewport_rows
        
        self.scroll_changed.emit(value)
        
        if self._loaded_row_count > 0:
            prefetch_threshold = self._loaded_row_count - (viewport_rows * self.PREFETCH_VIEWPORTS)
            if last_visible >= prefetch_threshold:
                self.prefetch_needed.emit(value)


# ==================== MAIN VIEWER ====================

class CacheViewer(QMainWindow):
    """Main window for viewing dataset cache."""

    def __init__(self, cache_dir: str = None):
        super().__init__()
        self.cache_dir = cache_dir
        self.setWindowTitle("Dataset Cache Viewer")
        self.setMinimumSize(1000, 700)
        
        # Data holders
        self.dataset = None
        self.statistics = None
        self.metadata = None
        self.vocab_data = []
        
        # Source filtering
        self.source_indices: Dict[str, List[int]] = {}
        self.language_indices: Dict[str, List[int]] = {}
        self.all_indices: List[int] = []
        self.current_source = "All"
        self.current_language = "All"
        
        # Lazy loading state
        self.model = LazyTableModel()
        self.worker: Optional[ContinuousLoaderWorker] = None
        self._loaded_blocks = 0
        self._tray_icon = None
        
        # Cache file paths
        self.cache_dataset_file = None
        self.cache_stats_file = None
        self.cache_metadata_file = None
        self.sentencepiece_vocab = None
        
        # Setup UI
        self._setup_ui()
        
        # JSONL data
        self.jsonl_data = {}  # {split_name: [records]}
        self.current_jsonl_split = None
        
        # Set cache directory
        if cache_dir:
            self._set_cache_dir(cache_dir)
        else:
            self._ask_cache_dir()
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header with cache path
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Cache Directory:"))
        self.cache_path_edit = QLineEdit()
        self.cache_path_edit.setReadOnly(True)
        header_layout.addWidget(self.cache_path_edit)
        
        self.change_btn = QPushButton("Change...")
        self.change_btn.clicked.connect(self._ask_cache_dir)
        header_layout.addWidget(self.change_btn)
        main_layout.addLayout(header_layout)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self._create_summary_tab()
        self._create_statistics_tab()
        self._create_samples_tab()
        self._create_vocabulary_tab()
        self._create_jsonl_tab()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
    def _ask_cache_dir(self):
        """Ask user to select cache directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Cache Directory",
            os.path.dirname(os.path.dirname(__file__)),
            QFileDialog.ShowDirsOnly
        )
        if folder:
            self._set_cache_dir(folder)
    
    def _set_cache_dir(self, path: str):
        """Set the cache directory and schedule data loading."""
        # Cancel any ongoing worker
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)
        
        self.cache_dir = path
        self.cache_path_edit.setText(path)
        
        # Set file paths
        self.cache_dataset_file = os.path.join(path, "prepared_dataset")
        self.cache_stats_file = os.path.join(path, "dataset_stats.pkl")
        self.cache_metadata_file = os.path.join(path, "cache_metadata.pkl")
        self.sentencepiece_vocab = os.path.join(path, "sentencepiece.vocab")
        
        # Reset state
        self.source_indices.clear()
        self.all_indices = []
        self.current_source = "All"
        self._loaded_blocks = 0
        
        # Schedule data loading after init completes (so tray icon is set)
        QTimer.singleShot(0, self._load_cache_data)
    
    # ==================== TAB 1: SUMMARY ====================
    
    def _create_summary_tab(self):
        """Create the general summary tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("Cache Summary")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Info group
        info_group = QGroupBox("Cache Information")
        info_layout = QFormLayout()
        
        self.cache_exists_label = QLabel()
        self.total_size_label = QLabel()
        self.last_modified_label = QLabel()
        
        info_layout.addRow("Cache Exists:", self.cache_exists_label)
        info_layout.addRow("Total Size:", self.total_size_label)
        info_layout.addRow("Last Modified:", self.last_modified_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Files table
        files_group = QGroupBox("Cache Files")
        files_layout = QVBoxLayout()
        
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(4)
        self.files_table.setHorizontalHeaderLabels(["File", "Size", "Modified", "Type"])
        self.files_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.files_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        files_layout.addWidget(self.files_table)
        
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)
        
        self.tab_widget.addTab(tab, "Summary")
    
    # ==================== TAB 2: STATISTICS ====================
    
    def _create_statistics_tab(self):
        """Create the statistics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("Dataset Statistics")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Stats group
        stats_group = QGroupBox("General Statistics")
        stats_layout = QFormLayout()
        
        self.total_samples_label = QLabel()
        self.avg_length_label = QLabel()
        self.min_length_label = QLabel()
        self.max_length_label = QLabel()
        
        stats_layout.addRow("Total Samples:", self.total_samples_label)
        stats_layout.addRow("Avg Text Length:", self.avg_length_label)
        stats_layout.addRow("Min Text Length:", self.min_length_label)
        stats_layout.addRow("Max Text Length:", self.max_length_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Source breakdown table
        source_group = QGroupBox("Samples by Source")
        source_layout = QVBoxLayout()
        
        self.source_table = QTableWidget()
        self.source_table.setColumnCount(2)
        self.source_table.setHorizontalHeaderLabels(["Source", "Samples"])
        self.source_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.source_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        source_layout.addWidget(self.source_table)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Language breakdown table
        language_group = QGroupBox("Samples by Language")
        language_layout = QVBoxLayout()
        
        self.language_stats_table = QTableWidget()
        self.language_stats_table.setColumnCount(3)
        self.language_stats_table.setHorizontalHeaderLabels(["Language", "Code", "Samples"])
        self.language_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.language_stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        language_layout.addWidget(self.language_stats_table)
        
        language_group.setLayout(language_layout)
        layout.addWidget(language_group)
        
        self.tab_widget.addTab(tab, "Statistics")
    
    # ==================== TAB 3: SAMPLES ====================
    
    def _create_samples_tab(self):
        """Create the samples tab with source filtering and virtual scrolling."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title row with source filter
        title_layout = QHBoxLayout()
        
        title = QLabel("Dataset Samples")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        
        # Source filter combo
        title_layout.addWidget(QLabel("Filter by Source:"))
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(150)
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        title_layout.addWidget(self.source_combo)
        
        # Language filter combo
        title_layout.addWidget(QLabel("Filter by Language:"))
        self.language_combo = QComboBox()
        self.language_combo.setMinimumWidth(100)
        self.language_combo.currentTextChanged.connect(self._on_language_changed)
        title_layout.addWidget(self.language_combo)
        
        layout.addLayout(title_layout)
        
        # Info row
        info_layout = QHBoxLayout()
        
        self.samples_count_label = QLabel("Loaded: 0 / 0")
        info_layout.addWidget(self.samples_count_label)
        
        info_layout.addStretch()
        
        self.source_info_label = QLabel("Source: All")
        info_layout.addWidget(self.source_info_label)
        
        layout.addLayout(info_layout)
        
        # Progress bar
        self.samples_progress = QProgressBar()
        self.samples_progress.setMaximumHeight(8)
        self.samples_progress.setTextVisible(False)
        layout.addWidget(self.samples_progress)
        
        # Splitter for table and detail
        splitter = QSplitter(Qt.Vertical)
        
        # Samples table (virtual scroll)
        self.samples_table = VirtualScrollTable()
        self.samples_table.setModel(self.model)
        self.samples_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.samples_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.samples_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.samples_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.samples_table.prefetch_needed.connect(self._on_prefetch_needed)
        self.samples_table.clicked.connect(self._on_sample_clicked)
        splitter.addWidget(self.samples_table)
        
        # Detail view
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        
        detail_label = QLabel("Full Text:")
        detail_label.setFont(QFont("Arial", 10, QFont.Bold))
        detail_layout.addWidget(detail_label)
        
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        detail_layout.addWidget(self.detail_text)
        
        splitter.addWidget(detail_widget)
        splitter.setSizes([400, 200])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(tab, "Samples")
    
    def _on_source_changed(self, source: str):
        """Handle source filter change."""
        self.current_source = source
        self._reload_with_filter()
    
    def _on_language_changed(self, language: str):
        """Handle language filter change."""
        self.current_language = language
        self._reload_with_filter()
    
    def _reload_with_filter(self):
        """Reload samples with current source and language filters."""
        # Cancel any ongoing worker
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)
        
        # Get indices for current filters
        if self.current_source == "All":
            indices = self.all_indices
        else:
            indices = self.source_indices.get(self.current_source, [])
        
        # Apply language filter
        if self.current_language != "All":
            lang_set = set(self.language_indices.get(self.current_language, []))
            indices = [i for i in indices if i in lang_set]
        
        # Reset model
        self.model.set_total_count(len(indices))
        self._loaded_blocks = 0
        
        # Update progress
        self.samples_progress.setMaximum(len(indices))
        self.samples_progress.setValue(0)
        self.samples_count_label.setText(f"Loaded: 0 / {len(indices):,}")
        self.source_info_label.setText(f"Source: {self.current_source} | Language: {self.current_language} ({len(indices):,} samples)")
        
        # Start loading first block
        if indices:
            self._load_samples_block(indices, 0)
    
    def _on_prefetch_needed(self, first_visible_row: int):
        """Handle prefetch request from scroll - load next block."""
        if self.worker and self.worker.isRunning():
            return
        
        if self.current_source == "All":
            indices = self.all_indices
        else:
            indices = self.source_indices.get(self.current_source, [])
        
        if self.current_language != "All":
            lang_set = set(self.language_indices.get(self.current_language, []))
            indices = [i for i in indices if i in lang_set]
        
        loaded = self.model.get_loaded_count()
        total = len(indices)
        
        if loaded >= total:
            return
        
        self._load_samples_block(indices, loaded)
    
    def _load_samples_block(self, indices: List[int], start_offset: int):
        """Load a block of samples in background thread."""
        if self.worker and self.worker.isRunning():
            return
        
        if self.worker:
            try:
                self.worker.block_ready.disconnect(self._on_block_ready)
                self.worker.error_occurred.disconnect(self._on_block_error)
            except (TypeError, RuntimeError):
                pass
        
        self.worker = ContinuousLoaderWorker(indices, self.dataset, start_offset, BLOCK_SIZE, getattr(self, '_token_lengths', None))
        self.worker.block_ready.connect(self._on_block_ready)
        self.worker.error_occurred.connect(self._on_block_error)
        self.worker.start()
    
    def _on_block_ready(self, items: List[Dict], offset: int, total: int):
        """Handle loaded block from worker."""
        self.model.add_block(items, offset)
        
        loaded = self.model.get_loaded_count()
        self.samples_count_label.setText(f"Loaded: {loaded:,} / {total:,}")
        self.samples_progress.setMaximum(total)
        self.samples_progress.setValue(loaded)
        
        # Update virtual scroll table with new loaded count
        self.samples_table.set_loaded_row_count(loaded)
        
        # Re-check if more prefetch needed (scroll position didn't change, so signal won't fire)
        if loaded < total:
            current_pos = self.samples_table.verticalScrollBar().value()
            self.samples_table._on_scroll(current_pos)
    
    def _on_block_error(self, error: str):
        """Handle worker error."""
        self.status_bar.showMessage(f"Error loading samples: {error}", 5000)
    
    def _on_sample_clicked(self, index: QModelIndex):
        """Handle sample click to show full text."""
        row = index.row()
        item = self.model.get_item(row)
        
        if item:
            text = item['text']
            source = item.get('source', '')
            language = item.get('language', '')
            parts = []
            if source:
                parts.append(f"Source: {source}")
            if language:
                parts.append(f"Language: {language}")
            if parts:
                header = ' | '.join(parts)
                self.detail_text.setPlainText(f"{header}\n\n{text}")
            else:
                self.detail_text.setPlainText(text)
        else:
            self.detail_text.setPlainText("Loading...")
    
    # ==================== TAB 4: VOCABULARY ====================
    
    def _create_vocabulary_tab(self):
        """Create the vocabulary tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("BPE Vocabulary")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Search controls
        search_layout = QHBoxLayout()
        
        search_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter token or ID...")
        self.search_input.textChanged.connect(self._filter_vocabulary)
        search_layout.addWidget(self.search_input)
        
        search_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Special Tokens", "Regular Tokens"])
        self.filter_combo.currentIndexChanged.connect(self._filter_vocabulary)
        search_layout.addWidget(self.filter_combo)
        
        self.vocab_count_label = QLabel()
        search_layout.addWidget(self.vocab_count_label)
        
        layout.addLayout(search_layout)
        
        # Vocabulary table
        self.vocab_table = QTableWidget()
        self.vocab_table.setColumnCount(3)
        self.vocab_table.setHorizontalHeaderLabels(["ID", "Token", "Score"])
        self.vocab_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vocab_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.vocab_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.vocab_table)
        
        self.tab_widget.addTab(tab, "Vocabulary")
    
    # ==================== TAB 5: JSONL ====================
    
    def _create_jsonl_tab(self):
        """Create the JSONL viewer tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Title
        title = QLabel("JSONL Export (GPT-2 Standard)")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Split selector
        selector_layout = QHBoxLayout()
        
        selector_layout.addWidget(QLabel("Split:"))
        self.jsonl_split_combo = QComboBox()
        self.jsonl_split_combo.addItems(["train", "val", "test"])
        self.jsonl_split_combo.currentTextChanged.connect(self._on_jsonl_split_changed)
        selector_layout.addWidget(self.jsonl_split_combo)
        
        selector_layout.addStretch()
        
        self.jsonl_count_label = QLabel("Records: 0")
        selector_layout.addWidget(self.jsonl_count_label)
        
        self.jsonl_size_label = QLabel("Size: 0 B")
        selector_layout.addWidget(self.jsonl_size_label)
        
        layout.addLayout(selector_layout)
        
        # Search/filter
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Search:"))
        self.jsonl_search_input = QLineEdit()
        self.jsonl_search_input.setPlaceholderText("Filter by text content...")
        self.jsonl_search_input.textChanged.connect(self._filter_jsonl)
        filter_layout.addWidget(self.jsonl_search_input)
        
        filter_layout.addWidget(QLabel("Show:"))
        self.jsonl_show_combo = QComboBox()
        self.jsonl_show_combo.addItems(["All", "Thinking", "Text Only", "Agent"])
        self.jsonl_show_combo.currentIndexChanged.connect(self._filter_jsonl)
        filter_layout.addWidget(self.jsonl_show_combo)
        
        layout.addLayout(filter_layout)
        
        # Splitter for table and detail
        splitter = QSplitter(Qt.Vertical)
        
        # JSONL table
        self.jsonl_table = QTableWidget()
        self.jsonl_table.setColumnCount(5)
        self.jsonl_table.setHorizontalHeaderLabels(["#", "Type", "Problem", "Thinking", "Answer"])
        self.jsonl_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.jsonl_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.jsonl_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.jsonl_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.jsonl_table.clicked.connect(self._on_jsonl_row_clicked)
        splitter.addWidget(self.jsonl_table)
        
        # Detail view
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        
        detail_tabs = QTabWidget()
        
        # Raw text tab
        self.jsonl_raw_text = QTextEdit()
        self.jsonl_raw_text.setReadOnly(True)
        self.jsonl_raw_text.setFont(QFont("Consolas", 10))
        detail_tabs.addTab(self.jsonl_raw_text, "Raw Text")
        
        # Parsed tab
        self.jsonl_parsed_text = QTextEdit()
        self.jsonl_parsed_text.setReadOnly(True)
        detail_tabs.addTab(self.jsonl_parsed_text, "Parsed")
        
        # Token IDs tab
        self.jsonl_tokens_text = QTextEdit()
        self.jsonl_tokens_text.setReadOnly(True)
        self.jsonl_tokens_text.setFont(QFont("Consolas", 10))
        detail_tabs.addTab(self.jsonl_tokens_text, "Token IDs")
        
        splitter.addWidget(detail_tabs)
        splitter.setSizes([400, 200])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(tab, "JSONL")
    
    def _on_jsonl_split_changed(self, split: str):
        """Handle JSONL split selection change."""
        self.current_jsonl_split = split
        self._load_jsonl_data(split)
    
    def _load_jsonl_data(self, split: str):
        """Load JSONL data for a specific split."""
        jsonl_dir = os.path.join(self.cache_dir, "jsonl")
        jsonl_file = os.path.join(jsonl_dir, f"{split}.jsonl")
        
        if not os.path.exists(jsonl_file):
            self.jsonl_count_label.setText("Records: 0")
            self.jsonl_size_label.setText("Size: N/A")
            self.jsonl_table.setRowCount(0)
            return
        
        try:
            # Get file size
            file_size = os.path.getsize(jsonl_file)
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.2f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.2f} MB"
            self.jsonl_size_label.setText(f"Size: {size_str}")
            
            # Load JSONL data
            self.jsonl_data[split] = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            import json
                            record = json.loads(line)
                            self.jsonl_data[split].append(record)
                        except json.JSONDecodeError:
                            continue
            
            total_records = len(self.jsonl_data[split])
            self.jsonl_count_label.setText(f"Records: {total_records:,}")
            
            # Populate table
            self._populate_jsonl_table(self.jsonl_data[split])
            
            self.status_bar.showMessage(
                f"JSONL {split} loaded: {total_records:,} records", 3000
            )
        
        except Exception as e:
            self.status_bar.showMessage(f"Error loading JSONL: {e}")
    
    def _populate_jsonl_table(self, records: List[Dict]):
        """Populate the JSONL table with records."""
        self.jsonl_table.setRowCount(len(records))
        
        for i, record in enumerate(records):
            text = record.get('text', '')
            
            # Parse the text to extract components
            record_type, problem, thinking, answer = self._parse_jsonl_record(text)
            
            # Row number
            row_item = QTableWidgetItem(str(i))
            row_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.jsonl_table.setItem(i, 0, row_item)
            
            # Type
            self.jsonl_table.setItem(i, 1, QTableWidgetItem(record_type))
            
            # Problem (truncated)
            problem_display = problem[:100] + "..." if len(problem) > 100 else problem
            self.jsonl_table.setItem(i, 2, QTableWidgetItem(problem_display))
            
            # Thinking (truncated, show first 50 chars)
            thinking_display = thinking[:50] + "..." if len(thinking) > 50 else thinking
            self.jsonl_table.setItem(i, 3, QTableWidgetItem(thinking_display))
            
            # Answer (truncated)
            answer_display = answer[:100] + "..." if len(answer) > 100 else answer
            self.jsonl_table.setItem(i, 4, QTableWidgetItem(answer_display))
    
    def _parse_jsonl_record(self, text: str) -> tuple:
        """Parse a JSONL record text into components."""
        record_type = "Text"
        problem = ""
        thinking = ""
        answer = ""
        
        # Try to parse GPT-2 standard format
        if '<|problem|>' in text:
            parts = text.split('<|problem|>')
            if len(parts) > 1:
                problem_part = parts[1]
                
                if '<|thinking|>' in problem_part:
                    # Thinking format: <|problem|>q<|thinking|>r<|final|>a
                    thinking_parts = problem_part.split('<|thinking|>')
                    problem = thinking_parts[0]
                    
                    if '<|final|>' in thinking_parts[1]:
                        final_parts = thinking_parts[1].split('<|final|>')
                        thinking = final_parts[0]
                        answer = final_parts[1]
                        record_type = "Thinking"
                    else:
                        thinking = thinking_parts[1]
                        record_type = "Thinking"
                
                elif '<|final|>' in problem_part:
                    # Text format: <|problem|>q<|final|>a
                    final_parts = problem_part.split('<|final|>')
                    problem = final_parts[0]
                    answer = final_parts[1]
                    record_type = "Text"
                
                elif '<|user|>' in problem_part:
                    # Agent format
                    record_type = "Agent"
                    problem = problem_part[:200]
                
                else:
                    problem = problem_part[:200]
        
        elif '<|user|>' in text:
            # Agent format
            record_type = "Agent"
            parts = text.split('<|user|>')
            if len(parts) > 1:
                problem = parts[1][:200]
        
        else:
            # Unknown format, show raw
            problem = text[:200]
        
        return record_type, problem, thinking, answer
    
    def _filter_jsonl(self):
        """Filter JSONL table based on search and type filter."""
        if not self.current_jsonl_split or self.current_jsonl_split not in self.jsonl_data:
            return
        
        records = self.jsonl_data[self.current_jsonl_split]
        search_text = self.jsonl_search_input.text().lower()
        filter_type = self.jsonl_show_combo.currentText()
        
        filtered = []
        for record in records:
            text = record.get('text', '')
            record_type, problem, thinking, answer = self._parse_jsonl_record(text)
            
            # Apply type filter
            if filter_type != "All":
                if filter_type == "Thinking" and record_type != "Thinking":
                    continue
                elif filter_type == "Text Only" and record_type != "Text":
                    continue
                elif filter_type == "Agent" and record_type != "Agent":
                    continue
            
            # Apply search filter
            if search_text:
                if search_text in text.lower():
                    filtered.append(record)
            else:
                filtered.append(record)
        
        self._populate_jsonl_table(filtered)
        self.jsonl_count_label.setText(f"Records: {len(filtered):,} / {len(records):,}")
    
    def _on_jsonl_row_clicked(self, index: QModelIndex):
        """Handle JSONL row click to show details."""
        row = index.row()
        
        if not self.current_jsonl_split or self.current_jsonl_split not in self.jsonl_data:
            return
        
        records = self.jsonl_data[self.current_jsonl_split]
        if row >= len(records):
            return
        
        record = records[row]
        text = record.get('text', '')
        
        # Raw text
        self.jsonl_raw_text.setPlainText(text)
        
        # Parsed view
        record_type, problem, thinking, answer = self._parse_jsonl_record(text)
        parsed_parts = []
        parsed_parts.append(f"Type: {record_type}")
        parsed_parts.append(f"\n--- Problem ---\n{problem}")
        if thinking:
            parsed_parts.append(f"\n--- Thinking ---\n{thinking}")
        parsed_parts.append(f"\n--- Answer ---\n{answer}")
        self.jsonl_parsed_text.setPlainText("\n".join(parsed_parts))
        
        # Token IDs (if present)
        token_ids = record.get('token_ids', [])
        if token_ids:
            tokens_str = ", ".join(str(t) for t in token_ids[:200])
            if len(token_ids) > 200:
                tokens_str += f"\n... ({len(token_ids)} total tokens)"
            self.jsonl_tokens_text.setPlainText(tokens_str)
        else:
            self.jsonl_tokens_text.setPlainText("No token_ids in this record")
    
    # ==================== DATA LOADING ====================
    
    def set_tray_icon(self, tray_icon):
        """Set the system tray icon reference for progress updates."""
        self._tray_icon = tray_icon
    
    def _load_cache_data(self):
        """Load all cache data with tray icon progress updates."""
        self.status_bar.showMessage("Loading cache data...")

        # Check if cache exists
        if not os.path.exists(self.cache_dir):
            self.status_bar.showMessage("Cache directory not found")
            QMessageBox.warning(
                self, "Warning",
                f"Cache directory not found:\n{self.cache_dir}"
            )
            return

        # Define loading phases: (name, method, weight)
        phases = [
            ("Loading summary...", self._load_summary_data, 1),
            ("Loading dataset...", self._init_samples, 5),
            ("Loading statistics...", self._load_statistics_data, 1),
            ("Loading vocabulary...", self._load_vocabulary, 1),
            ("Loading JSONL splits...", self._load_jsonl_splits, 1),
        ]
        total_weight = sum(w for _, _, w in phases)
        completed_weight = 0

        for i, (label, method, weight) in enumerate(phases):
            phase_base = int(completed_weight / total_weight * 100)
            phase_step = weight / total_weight * 100
            if self._tray_icon:
                self._tray_icon.update(phase_base, f"Phase {i+1}/{len(phases)}")
            try:
                def _cb(p_or_tuple, _desc="", _base=phase_base, _step=phase_step, _label=label, _i=i, _n=len(phases)):
                    if isinstance(p_or_tuple, tuple):
                        pct_val, desc = p_or_tuple
                    elif isinstance(_desc, str) and _desc:
                        pct_val = p_or_tuple
                        desc = _desc
                    else:
                        pct_val = p_or_tuple
                        desc = ""
                    pct = int(_base + pct_val * _step)
                    if self._tray_icon:
                        desc_text = f": {desc}" if desc else ""
                        self._tray_icon.update(pct, f"Phase {_i+1}/{_n}{desc_text}")
                method(progress_callback=_cb)
            except Exception as e:
                self.status_bar.showMessage(f"Error during load: {e}")
            completed_weight += weight

        # Set ready state on tray icon
        loaded = self.model.get_loaded_count()
        if self._tray_icon:
            self._tray_icon.set_ready(loaded)

        self.status_bar.showMessage("Cache loaded successfully", 3000)
    
    def _load_jsonl_splits(self, progress_callback=None):
        """Load available JSONL splits."""
        if progress_callback:
            progress_callback(0.0)

        jsonl_dir = os.path.join(self.cache_dir, "jsonl")
        
        if not os.path.exists(jsonl_dir):
            return
        
        # Find available splits
        available_splits = []
        for split in ["train", "val", "test"]:
            jsonl_file = os.path.join(jsonl_dir, f"{split}.jsonl")
            if os.path.exists(jsonl_file):
                available_splits.append(split)
        
        if available_splits:
            self.jsonl_split_combo.clear()
            self.jsonl_split_combo.addItems(available_splits)
            
            # Auto-load first split
            if available_splits:
                self._load_jsonl_data(available_splits[0])

        if progress_callback:
            progress_callback(1.0)
    
    def _load_summary_data(self, progress_callback=None):
        """Load summary tab data."""
        if progress_callback:
            progress_callback(0.0)

        # Check existence
        exists = (os.path.exists(self.cache_dataset_file) and
                  os.path.exists(self.cache_stats_file))
        self.cache_exists_label.setText("Yes" if exists else "No")
        self.cache_exists_label.setStyleSheet(
            f"color: {'green' if exists else 'red'}; font-weight: bold;"
        )
        
        # Calculate total size
        total_size = 0
        latest_mod = 0
        files_info = []
        
        all_files = []
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                all_files.append(os.path.join(root, file))

        total_files = len(all_files)
        for fi, filepath in enumerate(all_files):
            if progress_callback and fi % 50 == 0:
                progress_callback(fi / max(total_files, 1))
            rel_path = os.path.relpath(filepath, self.cache_dir)
            size = os.path.getsize(filepath)
            mod_time = os.path.getmtime(filepath)
            
            total_size += size
            latest_mod = max(latest_mod, mod_time)
            
            # Determine file type
            file_type = "Other"
            if file.endswith(".arrow"):
                file_type = "Arrow Data"
            elif file.endswith(".pkl"):
                file_type = "Pickle"
            elif file.endswith(".model"):
                file_type = "SentencePiece Model"
            elif file.endswith(".vocab"):
                file_type = "Vocabulary"
            elif file.endswith(".json"):
                file_type = "JSON Metadata"
            elif file.endswith(".jsonl"):
                file_type = "JSONL (GPT-2 Standard)"
            
            files_info.append({
                "name": rel_path,
                "size": size,
                "modified": mod_time,
                "type": file_type
            })

        if progress_callback:
            progress_callback(1.0)
        
        # Set total size
        if total_size < 1024:
            size_str = f"{total_size} B"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.2f} KB"
        else:
            size_str = f"{total_size / (1024 * 1024):.2f} MB"
        self.total_size_label.setText(size_str)
        
        # Set last modified
        if latest_mod > 0:
            mod_date = datetime.fromtimestamp(latest_mod).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            mod_date = "N/A"
        self.last_modified_label.setText(mod_date)
        
        # Populate files table
        self.files_table.setRowCount(len(files_info))
        for i, info in enumerate(files_info):
            self.files_table.setItem(i, 0, QTableWidgetItem(info["name"]))
            
            # Format size
            size = info["size"]
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.2f} MB"
            self.files_table.setItem(i, 1, QTableWidgetItem(size_str))
            
            # Format date
            mod_date = datetime.fromtimestamp(info["modified"]).strftime(
                "%Y-%m-%d %H:%M"
            )
            self.files_table.setItem(i, 2, QTableWidgetItem(mod_date))
            
            self.files_table.setItem(i, 3, QTableWidgetItem(info["type"]))
    
    def _load_statistics_data(self, progress_callback=None):
        """Load statistics tab data."""
        if progress_callback:
            progress_callback(0.0)

        if not os.path.exists(self.cache_stats_file):
            return
        
        try:
            with open(self.cache_stats_file, 'rb') as f:
                self.statistics = pickle.load(f)

            if progress_callback:
                progress_callback(0.5)
            
            # Set labels
            self.total_samples_label.setText(
                f"{self.statistics.get('total_samples', 0):,}"
            )
            self.avg_length_label.setText(
                f"{self.statistics.get('avg_text_length', 0):.1f} words"
            )
            self.min_length_label.setText(
                f"{self.statistics.get('min_text_length', 0)} words"
            )
            self.max_length_label.setText(
                f"{self.statistics.get('max_text_length', 0)} words"
            )
            
            # Populate source breakdown
            source_data = self.statistics.get('source_breakdown', {})
            self.source_table.setRowCount(len(source_data))
            
            for i, (source, count) in enumerate(sorted(source_data.items())):
                self.source_table.setItem(i, 0, QTableWidgetItem(source))
                self.source_table.setItem(i, 1, QTableWidgetItem(f"{count:,}"))
            
            # Populate language breakdown from dataset if available
            if self.dataset and 'language' in self.dataset.column_names:
                from collections import Counter
                all_langs = self.dataset['language']
                lang_counts = Counter(all_langs)
                
                # Language name mapping
                lang_names = {
                    'es': 'Spanish', 'en': 'English', 'fr': 'French', 'de': 'German',
                    'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'pl': 'Polish',
                    'cs': 'Czech', 'sv': 'Swedish', 'da': 'Danish', 'nb': 'Norwegian',
                    'fi': 'Finnish', 'el': 'Greek', 'hu': 'Hungarian', 'ro': 'Romanian',
                    'bg': 'Bulgarian', 'hr': 'Croatian', 'sk': 'Slovak', 'sl': 'Slovenian',
                    'lt': 'Lithuanian', 'lv': 'Latvian', 'et': 'Estonian', 'ga': 'Irish',
                    'ca': 'Catalan', 'gl': 'Galician', 'sq': 'Albanian', 'is': 'Icelandic',
                    'lb': 'Luxembourgish', 'mk': 'Macedonian', 'sr': 'Serbian', 'uk': 'Ukrainian',
                }
                
                self.language_stats_table.setRowCount(len(lang_counts))
                for i, (lang, count) in enumerate(lang_counts.most_common()):
                    name = lang_names.get(lang, lang)
                    self.language_stats_table.setItem(i, 0, QTableWidgetItem(name))
                    self.language_stats_table.setItem(i, 1, QTableWidgetItem(lang))
                    count_item = QTableWidgetItem(f"{count:,}")
                    count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.language_stats_table.setItem(i, 2, count_item)
            else:
                self.language_stats_table.setRowCount(1)
                self.language_stats_table.setItem(0, 0, QTableWidgetItem("No language data"))
                self.language_stats_table.setItem(0, 1, QTableWidgetItem("-"))
                self.language_stats_table.setItem(0, 2, QTableWidgetItem("-"))
        
        except Exception as e:
            self.status_bar.showMessage(f"Error loading statistics: {e}")
    
    def _init_samples(self, progress_callback=None):
        """Initialize samples tab with source indices and lazy loading."""
        from datasets import Dataset
        
        if not os.path.exists(self.cache_dataset_file):
            return
        
        try:
            if progress_callback:
                progress_callback(0.0, "Reading dataset from disk...")

            self.dataset = Dataset.load_from_disk(self.cache_dataset_file)
            total = len(self.dataset)

            if progress_callback:
                progress_callback(0.15, f"Dataset loaded ({total:,} rows)")
            
            # Pre-compute token counts (avoid loading token_ids per row)
            has_tokens = 'token_ids' in self.dataset.column_names
            if has_tokens:
                if progress_callback:
                    progress_callback(0.20, "Reading token_ids column...")
                all_token_ids = self.dataset['token_ids']
                if progress_callback:
                    progress_callback(0.30, "Computing token lengths...")
                self._token_lengths = [len(t) for t in all_token_ids]
            else:
                self._token_lengths = [0] * total

            if progress_callback:
                progress_callback(0.35, "Building source indices...")
            
            # Build source indices
            self.source_indices.clear()
            self.language_indices.clear()
            self.all_indices = list(range(total))
            
            has_source_column = 'source' in self.dataset.column_names
            has_language_column = 'language' in self.dataset.column_names
            
            if has_source_column:
                # Fast column access instead of row-by-row
                all_sources = self.dataset['source']
                
                sources_set = set(all_sources)
                self.source_indices = {s: [] for s in sources_set}
                
                for idx, source in enumerate(all_sources):
                    self.source_indices[source].append(idx)
                
                # Populate source combo (block signals to avoid spurious _reload_with_filter)
                self.source_combo.blockSignals(True)
                self.source_combo.clear()
                self.source_combo.addItem("All")
                for source in sorted(sources_set):
                    self.source_combo.addItem(source)
                self.source_combo.blockSignals(False)
            
            if progress_callback:
                progress_callback(0.55, "Building language indices...")

            if has_language_column:
                all_languages = self.dataset['language']
                
                languages_set = set(all_languages)
                self.language_indices = {l: [] for l in languages_set}
                
                for idx, lang in enumerate(all_languages):
                    self.language_indices[lang].append(idx)
                
                # Populate language combo (block signals to avoid spurious _reload_with_filter)
                self.language_combo.blockSignals(True)
                self.language_combo.clear()
                self.language_combo.addItem("All")
                for lang in sorted(languages_set):
                    self.language_combo.addItem(lang)
                self.language_combo.blockSignals(False)
                
                self.status_bar.showMessage(
                    f"Dataset loaded: {total:,} samples, "
                    f"{len(sources_set) if has_source_column else 0} sources, "
                    f"{len(languages_set)} languages", 3000
                )
            else:
                # No language column - show message
                self.language_combo.blockSignals(True)
                self.language_combo.clear()
                self.language_combo.addItem("All")
                self.language_combo.blockSignals(False)
            
            if not has_source_column:
                # No source column - show message
                self.source_combo.blockSignals(True)
                self.source_combo.clear()
                self.source_combo.addItem("All")
                self.source_combo.blockSignals(False)
                self.source_info_label.setText(
                    "Source: All (no source column in dataset)"
                )
            
            # Initialize model
            self.model.set_total_count(total)
            self.samples_progress.setMaximum(total)
            self.samples_progress.setValue(0)
            self.samples_count_label.setText(f"Loaded: 0 / {total:,}")

            if progress_callback:
                progress_callback(0.9)
            
            # Load first block synchronously so data is available immediately
            if total > 0:
                block_size = min(BLOCK_SIZE, total)
                items = []
                for i in range(block_size):
                    idx = self.all_indices[i]
                    row = self.dataset[idx]
                    items.append({
                        'index': idx,
                        'text': row.get('bpe_text', row.get('input_ids', '')),
                        'token_count': self._token_lengths[idx] if self._token_lengths and idx < len(self._token_lengths) else 0,
                        'source': row.get('source', 'Unknown'),
                        'language': row.get('language', '-')
                    })
                self.model.add_block(items, 0)
                self.samples_count_label.setText(f"Loaded: {block_size:,} / {total:,}")
                self.samples_progress.setValue(block_size)
                self.samples_table.set_loaded_row_count(block_size)

            if progress_callback:
                progress_callback(1.0)
        
        except Exception as e:
            self.status_bar.showMessage(f"Error loading samples: {e}")
    
    def _load_vocabulary(self, progress_callback=None):
        """Load vocabulary tab data."""
        if progress_callback:
            progress_callback(0.0)

        if not os.path.exists(self.sentencepiece_vocab):
            return
        
        try:
            self.vocab_data = []
            with open(self.sentencepiece_vocab, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if progress_callback and i % 500 == 0:
                        progress_callback(min(i / 8000, 0.9))
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        token, score = parts
                        try:
                            score_val = float(score)
                        except ValueError:
                            score_val = 0.0
                        
                        self.vocab_data.append({
                            'id': i,
                            'token': token,
                            'score': score_val
                        })
            
            self._update_vocab_table(self.vocab_data)

            if progress_callback:
                progress_callback(1.0)
        
        except Exception as e:
            self.status_bar.showMessage(f"Error loading vocabulary: {e}")
    
    def _filter_vocabulary(self):
        """Filter vocabulary based on search and filter combo."""
        search_text = self.search_input.text().lower()
        filter_type = self.filter_combo.currentText()
        
        filtered = []
        # GPT-2 standard special tokens
        special_tokens = {
            '<unk>', '<s>', '</s>',
            '<|problem|>', '<|thinking|>', '<|final|>',
            '<|user|>', '<|assistant|>',
            '<|system|>', '<|end|>', '<|sep|>',
            '<tool_call>', '</tool_call>', '<|tool_result|>',
        }
        for item in self.vocab_data:
            # Apply filter
            if filter_type == "Special Tokens":
                if item['token'] not in special_tokens:
                    continue
            elif filter_type == "Regular Tokens":
                if item['token'] in special_tokens:
                    continue
            
            # Apply search
            if search_text:
                if (search_text in item['token'].lower() or
                    search_text == str(item['id'])):
                    filtered.append(item)
            else:
                filtered.append(item)
        
        self._update_vocab_table(filtered)
    
    def _update_vocab_table(self, data: List[Dict]):
        """Update the vocabulary table with data."""
        self.vocab_table.setRowCount(len(data))
        
        for i, item in enumerate(data):
            # ID
            id_item = QTableWidgetItem(str(item['id']))
            id_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.vocab_table.setItem(i, 0, id_item)
            
            # Token
            token = item['token']
            self.vocab_table.setItem(i, 1, QTableWidgetItem(token))
            
            # Score
            score_item = QTableWidgetItem(f"{item['score']:.4f}")
            score_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.vocab_table.setItem(i, 2, score_item)
        
        self.vocab_count_label.setText(
            f"Showing {len(data)} / {len(self.vocab_data)} tokens"
        )
    
    def closeEvent(self, event):
        """Clean up worker thread and tray icon on close."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        if self._tray_icon:
            self._tray_icon.stop()
            self._tray_icon = None
        event.accept()


def main():
    """Run the cache viewer with optional system tray icon."""
    try:
        from APP_CACHE_VIEWER.system_tray import SystemTrayIcon
    except ModuleNotFoundError:
        from system_tray import SystemTrayIcon

    parser = argparse.ArgumentParser(
        description="Dataset Cache Viewer - Inspect cached dataset files"
    )
    parser.add_argument(
        "cache_dir",
        nargs="?",
        default=None,
        help="Path to dataset_cache directory"
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Run without system tray icon"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Create tray icon first (before window)
    tray_icon = None
    if not args.no_tray:
        tray_icon = SystemTrayIcon(
            on_quit=lambda: (tray_icon.stop(), app.quit()) if tray_icon else app.quit(),
        )
        if tray_icon.available:
            tray_icon.run()
        else:
            tray_icon = None

    # Create and show viewer
    viewer = CacheViewer(cache_dir=args.cache_dir)
    if tray_icon:
        viewer.set_tray_icon(tray_icon)
    viewer.show()

    exit_code = app.exec_()

    if tray_icon:
        tray_icon.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
