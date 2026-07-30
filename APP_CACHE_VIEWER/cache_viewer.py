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
                    'source': source
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
    
    def __init__(self, indices: List[int], dataset, block_size: int):
        super().__init__()
        self.indices = indices
        self.dataset = dataset
        self.block_size = block_size
        self._is_cancelled = False
        self._current_offset = 0
        self._lock = False
    
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
                    token_ids = row.get('token_ids', [])
                    source = row.get('source', 'Unknown')
                    
                    items.append({
                        'index': idx,
                        'text': text,
                        'token_count': len(token_ids),
                        'source': source
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
        self._headers = ["#", "Text", "Token Count", "Source"]
    
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
            
            # Placeholder for unloaded rows
            if col == 0:
                return str(row)
            elif col == 1:
                return "Loading..."
            elif col == 2:
                return "..."
            elif col == 3:
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
        self.all_indices: List[int] = []
        self.current_source = "All"
        
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
    
    def _reload_with_filter(self):
        """Reload samples with current source filter."""
        # Cancel any ongoing worker
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)
        
        # Get indices for current filter
        if self.current_source == "All":
            indices = self.all_indices
        else:
            indices = self.source_indices.get(self.current_source, [])
        
        # Reset model
        self.model.set_total_count(len(indices))
        self._loaded_blocks = 0
        
        # Update progress
        self.samples_progress.setMaximum(len(indices))
        self.samples_progress.setValue(0)
        self.samples_count_label.setText(f"Loaded: 0 / {len(indices):,}")
        self.source_info_label.setText(f"Source: {self.current_source} ({len(indices):,} samples)")
        
        # Start loading first block
        if indices:
            self._load_samples_block(indices, 0)
    
    def _on_scroll_bottom(self):
        """Handle scroll near bottom - load next block."""
        if self.current_source == "All":
            indices = self.all_indices
        else:
            indices = self.source_indices.get(self.current_source, [])
        
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
        
        self.worker = ContinuousLoaderWorker(indices, self.dataset, BLOCK_SIZE)
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
            self.detail_text.setPlainText(
                f"[Source: {source}]\n\n{text}"
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
        
        self.status_bar.showMessage("Cache loaded successfully", 3000)
    
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
            
            # Build source indices
            self.source_indices.clear()
            self.all_indices = list(range(total))
            
            has_source_column = 'source' in self.dataset.column_names
            
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
                
                self.status_bar.showMessage(
                    f"Dataset loaded: {total:,} samples, "
                    f"{len(sources_set)} sources", 3000
                )
            else:
                # No source column - show message
                self.source_combo.clear()
                self.source_combo.addItem("All")
                self.source_info_label.setText(
                    "Source: All (no source column in dataset)"
                )
                
                self.status_bar.showMessage(
                    f"Dataset loaded: {total:,} samples "
                    "(no source information available)", 3000
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
        for item in self.vocab_data:
            # Apply filter
            if filter_type == "Special Tokens":
                if item['token'] not in ['<unk>', '<s>', '</s>', '<thinking>', '</thinking>']:
                    continue
            elif filter_type == "Regular Tokens":
                if item['token'] in ['<unk>', '<s>', '</s>', '<thinking>', '</thinking>']:
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
