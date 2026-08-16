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
    Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex
)
from PyQt5.QtGui import QFont


# ==================== CONSTANTS ====================

BLOCK_SIZE = 500
SCROLL_THRESHOLD = 100


# ==================== WORKER THREAD ====================

class DatasetWorker(QThread):
    """Worker thread to load dataset blocks in background."""
    
    block_ready = pyqtSignal(list, int, int)  # items, start_idx, total
    error_occurred = pyqtSignal(str)
    
    def __init__(self, indices: List[int], dataset, block_size: int):
        """
        Initialize worker.
        
        Args:
            indices: List of dataset indices to load (filtered or all)
            dataset: HuggingFace Dataset object
            block_size: Number of items to load per block
        """
        super().__init__()
        self.indices = indices
        self.dataset = dataset
        self.block_size = block_size
        self._is_cancelled = False
    
    def cancel(self):
        self._is_cancelled = True
    
    def run(self):
        try:
            items = []
            total = len(self.indices)
            end_idx = min(self.block_size, total)
            
            for i in range(end_idx):
                if self._is_cancelled:
                    return
                
                idx = self.indices[i]
                row = self.dataset[idx]
                text = row.get('input_ids', row.get('bpe_text', ''))
                token_ids = row.get('token_ids', [])
                source = row.get('source', 'Unknown')
                
                items.append({
                    'index': idx,
                    'text': text,
                    'token_count': len(token_ids),
                    'source': source,
                    'language': row.get('language', '-')
                })
            
            if not self._is_cancelled:
                self.block_ready.emit(items, 0, total)
        
        except Exception as e:
            if not self._is_cancelled:
                self.error_occurred.emit(str(e))


# ==================== CONTINUOUS LOADER WORKER ====================

class ContinuousLoaderWorker(QThread):
    """Worker thread for continuous loading as user scrolls."""
    
    block_ready = pyqtSignal(list, int, int)  # items, offset, total
    error_occurred = pyqtSignal(str)
    loading_finished = pyqtSignal()
    
    def __init__(self, indices: List[int], dataset, block_size: int, token_lengths: List[int] = None):
        super().__init__()
        self.indices = indices
        self.dataset = dataset
        self.block_size = block_size
        self._is_cancelled = False
        self._current_offset = 0
        self._lock = False
        self._token_lengths = token_lengths or []
    
    def cancel(self):
        self._is_cancelled = True
    
    def start_loading(self, from_offset: int = 0):
        """Start or resume loading from offset."""
        if not self._lock:
            self._current_offset = from_offset
            self._lock = True
            if not self.isRunning():
                self.start()
    
    def run(self):
        try:
            total = len(self.indices)
            
            while self._current_offset < total and not self._is_cancelled:
                items = []
                end_idx = min(self._current_offset + self.block_size, total)
                
                for i in range(self._current_offset, end_idx):
                    if self._is_cancelled:
                        return
                    
                    idx = self.indices[i]
                    row = self.dataset[idx]
                    text = row.get('input_ids', row.get('bpe_text', ''))
                    source = row.get('source', 'Unknown')
                    
                    # Use pre-computed token length
                    token_count = self._token_lengths[idx] if idx < len(self._token_lengths) else 0
                    
                    items.append({
                        'index': idx,
                        'text': text,
                        'token_count': token_count,
                        'source': source,
                        'language': row.get('language', '-')
                    })
                
                if items and not self._is_cancelled:
                    self.block_ready.emit(items, self._current_offset, total)
                
                self._current_offset = end_idx
            
            self._lock = False
            if not self._is_cancelled:
                self.loading_finished.emit()
        
        except Exception as e:
            self._lock = False
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
        """Add a block of loaded items."""
        first_row = start_idx
        last_row = min(start_idx + len(items), self._total_count) - 1
        
        if first_row > last_row:
            return
        
        self.beginInsertRows(QModelIndex(), first_row, last_row)
        
        for i, item in enumerate(items):
            idx = start_idx + i
            if idx < len(self._data):
                self._data[idx] = item
        
        self.endInsertRows()
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
    """QTableView with virtual scroll signal."""
    
    scroll_near_bottom = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.verticalScrollBar().valueChanged.connect(self._check_scroll)
    
    def _check_scroll(self, value):
        scrollbar = self.verticalScrollBar()
        if scrollbar.maximum() - value < SCROLL_THRESHOLD:
            self.scroll_near_bottom.emit()


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
        """Set the cache directory and load data."""
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
        
        # Load data
        self._load_cache_data()
    
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
        self.samples_table.scroll_near_bottom.connect(self._on_scroll_bottom)
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
    
    def _on_scroll_bottom(self):
        """Handle scroll near bottom - load next block."""
        if self.current_source == "All":
            indices = self.all_indices
        else:
            indices = self.source_indices.get(self.current_source, [])
        
        # Apply language filter
        if self.current_language != "All":
            lang_set = set(self.language_indices.get(self.current_language, []))
            indices = [i for i in indices if i in lang_set]
        
        loaded = self.model.get_loaded_count()
        total = len(indices)
        
        if loaded >= total:
            return
        
        # Load next block
        self._load_samples_block(indices, loaded)
    
    def _load_samples_block(self, indices: List[int], start_offset: int):
        """Load a block of samples in background thread."""
        if self.worker and self.worker.isRunning():
            # Don't cancel if it's already loading from same offset
            return
        
        self.worker = ContinuousLoaderWorker(indices, self.dataset, BLOCK_SIZE, getattr(self, '_token_lengths', None))
        self.worker.block_ready.connect(self._on_block_ready)
        self.worker.error_occurred.connect(self._on_block_error)
        self.worker.loading_finished.connect(self._on_loading_finished)
        self.worker.start_loading(start_offset)
    
    def _on_block_ready(self, items: List[Dict], offset: int, total: int):
        """Handle loaded block from worker."""
        self.model.add_block(items, offset)
        
        loaded = self.model.get_loaded_count()
        self.samples_count_label.setText(f"Loaded: {loaded:,} / {total:,}")
        self.samples_progress.setMaximum(total)
        self.samples_progress.setValue(loaded)
    
    def _on_block_error(self, error: str):
        """Handle worker error."""
        self.status_bar.showMessage(f"Error loading samples: {error}", 5000)
    
    def _on_loading_finished(self):
        """Handle loading completion."""
        loaded = self.model.get_loaded_count()
        self.status_bar.showMessage(f"All {loaded:,} samples loaded", 3000)
    
    def _on_sample_clicked(self, index: QModelIndex):
        """Handle sample click to show full text."""
        row = index.row()
        item = self.model.get_item(row)
        
        if item:
            text = item['text']
            source = item.get('source', 'Unknown')
            language = item.get('language', '-')
            self.detail_text.setPlainText(
                f"[Source: {source} | Language: {language}]\n\n{text}"
            )
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
    
    def _load_cache_data(self):
        """Load all cache data."""
        self.status_bar.showMessage("Loading cache data...")
        
        # Check if cache exists
        if not os.path.exists(self.cache_dir):
            self.status_bar.showMessage("Cache directory not found")
            QMessageBox.warning(
                self, "Warning",
                f"Cache directory not found:\n{self.cache_dir}"
            )
            return
        
        # Load all data
        self._load_summary_data()
        self._load_statistics_data()
        self._init_samples()
        self._load_vocabulary()
        self._load_jsonl_splits()
        
        self.status_bar.showMessage("Cache loaded successfully", 3000)
    
    def _load_jsonl_splits(self):
        """Load available JSONL splits."""
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
    
    def _load_summary_data(self):
        """Load summary tab data."""
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
        
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                filepath = os.path.join(root, file)
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
    
    def _load_statistics_data(self):
        """Load statistics tab data."""
        if not os.path.exists(self.cache_stats_file):
            return
        
        try:
            with open(self.cache_stats_file, 'rb') as f:
                self.statistics = pickle.load(f)
            
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
    
    def _init_samples(self):
        """Initialize samples tab with source indices and lazy loading."""
        from datasets import Dataset
        
        if not os.path.exists(self.cache_dataset_file):
            return
        
        try:
            self.dataset = Dataset.load_from_disk(self.cache_dataset_file)
            total = len(self.dataset)
            
            # Pre-compute token counts (avoid loading token_ids per row)
            has_tokens = 'token_ids' in self.dataset.column_names
            if has_tokens:
                all_token_ids = self.dataset['token_ids']
                self._token_lengths = [len(t) for t in all_token_ids]
            else:
                self._token_lengths = [0] * total
            
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
                
                # Populate source combo
                self.source_combo.clear()
                self.source_combo.addItem("All")
                for source in sorted(sources_set):
                    self.source_combo.addItem(source)
            
            if has_language_column:
                all_languages = self.dataset['language']
                
                languages_set = set(all_languages)
                self.language_indices = {l: [] for l in languages_set}
                
                for idx, lang in enumerate(all_languages):
                    self.language_indices[lang].append(idx)
                
                # Populate language combo
                self.language_combo.clear()
                self.language_combo.addItem("All")
                for lang in sorted(languages_set):
                    self.language_combo.addItem(lang)
                
                self.status_bar.showMessage(
                    f"Dataset loaded: {total:,} samples, "
                    f"{len(sources_set) if has_source_column else 0} sources, "
                    f"{len(languages_set)} languages", 3000
                )
            else:
                # No language column - show message
                self.language_combo.clear()
                self.language_combo.addItem("All")
            
            if not has_source_column:
                # No source column - show message
                self.source_combo.clear()
                self.source_combo.addItem("All")
                self.source_info_label.setText(
                    "Source: All (no source column in dataset)"
                )
            
            # Initialize model
            self.model.set_total_count(total)
            self.samples_progress.setMaximum(total)
            self.samples_progress.setValue(0)
            self.samples_count_label.setText(f"Loaded: 0 / {total:,}")
            
            # Load first block
            if total > 0:
                self._load_samples_block(self.all_indices, 0)
        
        except Exception as e:
            self.status_bar.showMessage(f"Error loading samples: {e}")
    
    def _load_vocabulary(self):
        """Load vocabulary tab data."""
        if not os.path.exists(self.sentencepiece_vocab):
            return
        
        try:
            self.vocab_data = []
            with open(self.sentencepiece_vocab, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
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
            '<tool_call>', '</tool_call>', '<|tool_result|>',
            '<thinking>', '</thinking>',
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
        """Clean up worker thread on close."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        event.accept()


def main():
    """Run the cache viewer."""
    parser = argparse.ArgumentParser(
        description="Dataset Cache Viewer - Inspect cached dataset files"
    )
    parser.add_argument(
        "cache_dir",
        nargs="?",
        default=None,
        help="Path to dataset_cache directory"
    )
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    viewer = CacheViewer(cache_dir=args.cache_dir)
    viewer.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
