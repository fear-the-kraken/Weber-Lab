#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom widgets and functions used in main P-wave annotation window

@author: fearthekraken
"""
import numpy as np
import matplotlib.colors as mcolors
import pyautogui
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import pdb


def px_w(width, screen_width, ref_width=1920):
    """
    Convert pixel width from reference computer monitor to current screen
    @Params
    width - integer pixel width to convert
    screen_width - width of current monitor in pixels
    ref_width - width of reference monitor; default value is from Weber lab computer
    @Returns
    new_width - proportional pixel width on current screen
    """
    new_width = int(round(screen_width * (width/ref_width)))
    new_width = min(max(1,new_width), screen_width)
    return new_width


def px_h(height, screen_height, ref_height=1080):
    """
    Scale pixel height from reference computer monitor to current screen
    """
    new_height = int(round(screen_height * (height/ref_height)))
    new_height = min(max(1,new_height), screen_height)
    return new_height


def vline(orientation='v'):
    """
    Create vertical or horizontal line to separate widget containers
    @Returns
    line - QFrame object
    """
    line = QtWidgets.QFrame()
    if orientation == 'v':
        line.setFrameShape(QtWidgets.QFrame.VLine)
    elif orientation == 'h':
        line.setFrameShape(QtWidgets.QFrame.HLine)
    else:
        print('###   INVALID LINE ORIENTATION   ###')
        return
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    line.setLineWidth(3)
    line.setMidLineWidth(3)
    line.setContentsMargins(0,15,0,15)
    return line


def pg_symbol(symbol, font=QtGui.QFont('San Serif')):
    """
    Create custom pyqtgraph marker from keyboard character
    @Params
    symbol - single character (e.g. @) to convert into plot marker
    font - character font
    @Returns
    mapped_symbol - customized marker for pyqtgraph plots
    """
    # max of one character for symbol
    assert len(symbol) == 1
    pg_symbol = QtGui.QPainterPath()
    pg_symbol.addText(0, 0, font, symbol)
    # scale symbol
    br = pg_symbol.boundingRect()
    scale = min(1. / br.width(), 1. / br.height())
    tr = QtGui.QTransform()
    tr.scale(scale, scale)
    # center symbol in bounding box
    tr.translate(-br.x() - br.width() / 2., -br.y() - br.height() / 2.)
    mapped_symbol = tr.map(pg_symbol)
    return mapped_symbol


def warning_dlg(msg):
    """
    Execute dialog box with yes/no options
    @Returns
    1 ("yes") or 0 ("no")
    """
    # create dialog box and yes/no buttons
    dlg = QtWidgets.QDialog()
    font = QtGui.QFont()
    font.setPointSize(12)
    dlg.setFont(font)
    msg = QtWidgets.QLabel(msg)
    msg.setAlignment(QtCore.Qt.AlignCenter)
    QBtn = QtWidgets.QDialogButtonBox.Yes | QtWidgets.QDialogButtonBox.No
    bbox = QtWidgets.QDialogButtonBox(QBtn)
    bbox.accepted.connect(dlg.accept)
    bbox.rejected.connect(dlg.reject)
    # set layout, run window
    lay = QtWidgets.QVBoxLayout()
    lay.addWidget(msg)
    lay.addWidget(bbox)
    dlg.setLayout(lay)
    if dlg.exec():
        return 1
    else:
        return 0


def warning_msgbox(msg):
    """
    Execute message box with yes/no/cancel options
    @Returns
    1 ("yes"), 0 ("no"), or -1 ("cancel")
    """
    # create message box and yes/no/cancel options
    dlg = QtWidgets.QMessageBox()
    dlg.setText(msg)
    dlg.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    dlg.setDefaultButton(QtWidgets.QMessageBox.No)
    res = dlg.exec()
    if res == QtWidgets.QMessageBox.Yes:
        return 1
    elif res == QtWidgets.QMessageBox.No:
        return 0
    else:
        return -1


def back_next_btns(parent, name):
    """
    Create "back" or "next" arrow buttons for viewing different sets of plotting options
    @Params
    parent - main annotation window
    name - string defining object name and symbol; includes either "back" or "next"
    @Returns
    btn - green QPushButton object with left or right-facing arrow
    """
    # determine button size
    wpx, hpx = pyautogui.size()
    bsize = px_w(20, wpx)
    isize = px_w(17, wpx)
    # set button properties
    btn = QtWidgets.QPushButton()
    btn.setObjectName(name)
    btn.setFixedSize(bsize,bsize)
    if 'back' in name:
        btn.setIcon(parent.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack))
    elif 'next' in name:
        btn.setIcon(parent.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward))
    btn.setIconSize(QtCore.QSize(isize,isize))
    btn.setStyleSheet('QPushButton {border : none; margin-top : 2px}')
    return btn


def back_next_event(parent, name):
    """
    Create "back" or "next" arrow buttons for viewing each artifact in live data plot
    @Params
    parent - main annotation window
    name - string defining object name and symbol; includes either "back" or "next"
    @Returns
    btn - black & white QPushButton object with left or right-facing arrow
    """
    # determine button size
    wpx, hpx = pyautogui.size()
    bw, bh = [px_w(25, wpx), px_h(15, hpx)]
    iw, ih = [px_w(22, wpx), px_h(13, hpx)]
    # set button properties
    btn = QtWidgets.QPushButton()
    btn.setObjectName(name)
    btn.setFixedSize(bw,bh)
    if 'back' in name:
        btn.setIcon(parent.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward))
    elif 'next' in name:
        btn.setIcon(parent.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekForward))
    btn.setIconSize(QtCore.QSize(iw,ih))
    return btn
    

def show_hide_event(parent):
    """
    Create toggle button for showing and hiding each artifact in live data plot
    @Params
    parent - main annotation window
    @Returns
    btn - QPushButton object with show/hide icons for checked/unchecked states
    """
    # determine button size
    wpx, hpx = pyautogui.size()
    bsize = px_w(25, wpx)
    radius = px_w(5, wpx)
    # set button properties
    btn = QtWidgets.QPushButton()
    btn.setCheckable(True)
    btn.setChecked(False)
    btn.setFixedSize(bsize,bsize)
    btn.setStyleSheet('QPushButton'
                      '{ border : 2px solid gray;'
                      'border-style : outset;'
                      f'border-radius : {radius}px;'
                      'padding : 1px;'
                      'background-color : lightgray;'
                      'image : url("icons/hide_icon.png") }'
                      'QPushButton:checked'
                      '{ background-color : white;'
                      'image : url("icons/show_icon.png") }')
    return btn
    

def update_noise_btn(top_parent, parent, icon):
    """
    Create checkable buttons for handling noise detection and visualization
    @Params
    top_parent - main annotation window
    parent - noise widget in main window
    icon - if "save": update current noise indices with newly detected noise
           if "reset": replace current noise indices with newly detected noise
           if "calc": calculate/show EEG spectrogram with EEG noise excluded
    @Returns
    btn - QPushButton object with specified icon
    """
    # determine button size
    wpx, hpx = pyautogui.size()
    bsize = px_w(28, wpx)
    isize = px_w(18, wpx)
    # set button properties
    btn = QtWidgets.QPushButton(parent)
    btn.setCheckable(True)
    btn.setFixedSize(bsize,bsize)
    if icon == 'save':
        btn.setIcon(top_parent.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
    elif icon == 'reset':
        btn.setIcon(top_parent.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
    elif icon == 'calc':
        btn.setIcon(top_parent.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView))
    btn.setIconSize(QtCore.QSize(isize,isize))  # 28,18
    btn.setStyleSheet('QPushButton:checked {background-color : rgb(200,200,200)}')
    return btn


class FreqBandWindow(QtWidgets.QDialog):
    """
    Pop-up window for setting frequency band name, range, and plotting color
    """
    def __init__(self):
        super(FreqBandWindow, self).__init__()
        wpx, hpx = pyautogui.size()
        # get set of widths and heights, standardized by monitor dimensions
        wspace5, wspace10, wspace15, wspace20 = [px_w(w, wpx) for w in [5,10,15,20]]
        hspace5, hspace10, hspace15, hspace20 = [px_h(h, hpx) for h in [5,10,15,20]]
        # set contents margins, central layout
        self.setContentsMargins(wspace10,hspace10,wspace10,hspace10)
        self.centralLayout = QtWidgets.QVBoxLayout(self)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.setFont(font)
        
        # band name
        row1 = QtWidgets.QVBoxLayout()
        row1.setSpacing(hspace10)
        name_label = QtWidgets.QLabel('Band name')
        name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.name_input = QtWidgets.QLineEdit()
        row1.addWidget(name_label)
        row1.addWidget(self.name_input)
        
        # min and max frequencies
        row2 = QtWidgets.QVBoxLayout()
        row2.setSpacing(hspace10)
        range_label = QtWidgets.QLabel('Frequency Range')
        range_label.setAlignment(QtCore.Qt.AlignCenter)
        freq_row = QtWidgets.QHBoxLayout()
        freq_row.setSpacing(wspace5)
        self.freq1_input = QtWidgets.QDoubleSpinBox()
        self.freq1_input.setMaximum(500)
        self.freq1_input.setDecimals(1)
        self.freq1_input.setSingleStep(0.5)
        self.freq1_input.setSuffix(' Hz')
        self.freq1_input.valueChanged.connect(self.enable_save)
        freq_dash = QtWidgets.QLabel('-')
        self.freq2_input = QtWidgets.QDoubleSpinBox()
        self.freq2_input.setMaximum(500)
        self.freq2_input.setDecimals(1)
        self.freq2_input.setSingleStep(0.5)
        self.freq2_input.setSuffix(' Hz')
        self.freq2_input.valueChanged.connect(self.enable_save)
        freq_row.addWidget(self.freq1_input)
        freq_row.addWidget(freq_dash)
        freq_row.addWidget(self.freq2_input)
        row2.addWidget(range_label)
        row2.addLayout(freq_row)
        
        # plotting color
        row3 = QtWidgets.QVBoxLayout()
        row3.setSpacing(hspace5)
        color_label = QtWidgets.QLabel('Color')
        color_label.setAlignment(QtCore.Qt.AlignCenter)
        self.color_input = QtWidgets.QComboBox()
        # get list of CSS colors sorted by name and hue
        colors = mcolors.CSS4_COLORS
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) \
                        for name,color in colors.items())
        colornames = [name for hsv, name in by_hsv]
        for cname in colornames:
            pixmap = QtGui.QPixmap(100,100)
            pixmap.fill(QtGui.QColor(cname))
            self.color_input.addItem(QtGui.QIcon(pixmap), cname)
        row3.addWidget(color_label)
        row3.addWidget(self.color_input)
        
        # save or cancel changes
        self.bbox = QtWidgets.QDialogButtonBox()
        self.save_btn = self.bbox.addButton(QtWidgets.QDialogButtonBox.Save)
        self.save_btn.setDisabled(True)
        self.cancel_btn = self.bbox.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.bbox.setCenterButtons(True)
        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)
        
        # add widgets
        self.centralLayout.addLayout(row1)
        self.centralLayout.addLayout(row2)
        self.centralLayout.addLayout(row3)
        self.centralLayout.addWidget(self.bbox)
        self.centralLayout.setSpacing(hspace20)
    
    def enable_save(self):
        """
        Allow parameters to be saved if frequency range is valid
        """
        # max band frequency must be greater than min frequency
        f1, f2 = self.freq1_input.value(), self.freq2_input.value()
        self.save_btn.setEnabled(f2 > f1)
        
    
class FreqBandLabel(pg.LabelItem):
    """
    Text label for currently plotted EEG frequency band power
    """
    def __init__(self, parent=None):
        super(FreqBandLabel, self).__init__()
        self.mainWin = parent
        
    def contextMenuEvent(self, event):
        """
        Set context menu to edit, delete, or add new frequency band
        """
        self.menu = QtWidgets.QMenu()
        # edit freq band properties
        editAction = QtWidgets.QAction('Edit band')
        editAction.setObjectName('edit')
        editAction.triggered.connect(self.mainWin.freq_band_window)
        self.menu.addAction(editAction)
        # delete freq band
        deleteAction = QtWidgets.QAction('Delete band')
        deleteAction.setObjectName('delete')
        deleteAction.triggered.connect(self.mainWin.freq_band_window)
        self.menu.addAction(deleteAction)
        # add new freq band
        addAction = QtWidgets.QAction('Add new band')
        addAction.setObjectName('add')
        addAction.triggered.connect(self.mainWin.freq_band_window)
        self.menu.addAction(addAction)
        self.menu.exec_(QtGui.QCursor.pos())
            
        
    def set_info(self, freq1, freq2, band_name='', color='white'):
        """
        Update label of the current frequency band plot
        @Params
        freq1, freq2 - min and max frequencies in band
        band_name - name of frequency band (e.g. delta, theta, sigma...)
        color - color of band power plot
        """
        txt = f'{freq1} - {freq2} Hz'
        if band_name:
            txt = band_name + ' (' + txt + ')'
        labelOpts = {'color':color, 'size':'12pt'}
        self.setText(txt, **labelOpts)

    
class PlotButton(QtWidgets.QPushButton):
    """
    Data plotting button; instantiates pop-up figure window for specific graph
    """
    def __init__(self, parent, name, color, reqs=[]):
        super(PlotButton, self).__init__()
        self.mainWin = parent
        self.base_name = name
        self.color = color
        self.reqs = reqs
        
        # determine button size
        wpx, hpx = pyautogui.size()
        bsize = px_w(20, wpx)
        self.radius = int(bsize/2)
        self.bwidth = int(bsize/5)
        self.setFixedSize(bsize,bsize)
        # set button style
        self.setStyleSheet('QPushButton'
                           '{ border-color : gray;'
                           'border-width : ' + str(self.bwidth) + 'px;'
                           'border-style : outset;'
                           'border-radius : ' + str(self.radius) + 'px;'
                           'padding : 1px;'
                           'background-color : ' + self.color + ' }'
                           'QPushButton:pressed'
                           '{ border-width : ' + str(self.bwidth) + 'px;'
                           'background-color : rgba(139,58,58,255) }'
                           'QPushButton:disabled'
                           '{ border-width : ' + str(self.bwidth*3) + 'px;'
                           'background-color : ' + self.color + ' }')
        self.name = str(self.base_name)
    
    
    def enable_btn(self):
        """
        Enable button if recording has all required plot elements
        """
        enable = True
        if 'hasEMG' in self.reqs and self.mainWin.hasEMG == False:
            enable = False
        if 'hasDFF' in self.reqs and self.mainWin.hasDFF == False:
            enable = False
        if 'recordPwaves' in self.reqs and self.mainWin.hasPwaves == False:
            enable = False
        if 'lsrTrigPwaves' in self.reqs and self.mainWin.lsrTrigPwaves == False:
            enable = False
        if 'opto' in self.reqs and self.mainWin.optoMode == '':
            enable = False
        if 'ol' in self.reqs and self.mainWin.optoMode != 'ol':
            enable = False
        if 'cl' in self.reqs and self.mainWin.optoMode != 'cl':
            enable = False
        self.setEnabled(enable)
    
    
    def single_mode(self, p_event):
        """
        Change appearance when single event is selected
        """
        if p_event == 'lsr_pi':
            c = 'rgba(0,0,255,200)'      # blue background
        elif p_event == 'spon_pi':
            c = 'rgba(255,255,255,200)'  # white background
        elif p_event == 'lsr':
            c = 'rgba(17,107,41,200)'    # green background
        elif p_event == 'other':
            c = 'rgba(104,15,122,200)'   # purple background
        
        if self.isEnabled():
            # update plot button style
            self.setStyleSheet('QPushButton'
                               '{ border-color : rgba(255,0,0,215);'
                               f'border-width : {self.bwidth}px;'
                               'border-style : outset;'
                               f'border-radius : {int(round(self.bwidth/2))}px;'
                               'padding : 1px;'
                               'background-color : ' + c + ' }'
                               'QPushButton:pressed'
                               '{ background-color : rgba(255,255,0,255) }')
            self.name = 'Single ' + str(self.base_name)
    
    
    def avg_mode(self):
        """
        Reset appearance when single event is deselected
        """
        if self.isEnabled():
            # reset plot button style when selected point is cleared
            self.setStyleSheet('QPushButton'
                               '{ border-color : gray;'
                               f'border-width : {self.bwidth}px;'
                               'border-style : outset;'
                               f'border-radius : {self.radius}px;'
                               'padding : 1px;'
                               'background-color : ' + self.color + ' }'
                               'QPushButton:pressed'
                               '{ background-color : rgba(139,58,58,255) }')
            self.name = str(self.base_name)


class ArrowButton(pg.ArrowItem):
    """
    Arrow button representing the starting or ending point of a user-selected 
    data sequence in a GraphEEG object
    
    An ArrowButton is black when not selected; if its corresponding point is not 
    in the currently plotted time window, clicking the button will bring the point
    to the center of the graph. An additional click will "select" the arrow button,
    changing the color to white and determining whether the next double-clicked point 
    will update the starting or ending index of the selected sequence
    
    Example: Suppose the starting index of a currently selected sequence is 100, and
             the ending index is 200, so the sequence is [100, 101, 102 ... 199, 200].
             * If the user selects the start ArrowButton, the next double-clicked point
               (e.g. 150) will replace the current starting index, so the new selected
               sequence is [150, 151, 152 ... 199, 200]
             * If the user selects the end ArrowButton, the next double-clicked point
               (e.g. 150) will replace the current ending index, so the new selected
               sequence is [100, 101, 102 ... 149, 150]
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mainWin = parent
        # set arrow style
        wpx, hpx = pyautogui.size()
        hl, tl, tw = [px_w(n, wpx) for n in [17,17,6]]
        opts = {'angle': -90, 'headLen':hl, 'tipAngle':45, 'tailLen':tl, 'tailWidth':tw, 
                'pen':pg.mkPen((255,255,255),width=2), 'brush':pg.mkBrush((0,0,0))}
        self.setStyle(**opts)
        
        self.active = False   # initial button state is "unselected"
        self.pressPos = None  # clear mouse press position
        
    def mousePressEvent(self, event):
        """
        Save location of user mouse click
        """
        self.pressPos = event.pos()
        
    def mouseReleaseEvent(self, event):
        """
        Update ArrowButton selection in main window
        """
        if self.pressPos is not None and event.pos() in self.boundingRect():
            self.mainWin.switch_noise_boundary(self)
        self.pressPos = None
            
    def update_style(self):
        """
        Change color to white when selected, or black when unselected
        """
        if self.active == True:
            self.setBrush(pg.mkBrush((255,255,255)))
        elif self.active == False:
            self.setBrush(pg.mkBrush((0,0,0)))


class HelpWindow(QtWidgets.QDialog):
    """
    Informational pop-up window listing possible options for user keyboard input
    """
    def __init__(self, parent=None):
        super(HelpWindow, self).__init__(parent)
        wpx, hpx = pyautogui.size()
        
        # set contents margins, central layout
        cm = px_w(25, wpx)
        self.setContentsMargins(cm,cm,cm,cm)
        self.centralLayout = QtWidgets.QVBoxLayout(self)
        
        # set fonts
        subheaderFont = QtGui.QFont()
        subheaderFont.setPointSize(12)
        subheaderFont.setBold(True)
        subheaderFont.setUnderline(True)
        keyFont = QtGui.QFont()
        keyFont.setFamily('Courier')
        keyFont.setPointSize(18)
        keyFont.setBold(True)
        font = QtGui.QFont()
        font.setPointSize(12)
        # get set of widths and heights, standardized by monitor dimensions
        wspace5, wspace10, wspace15, wspace20 = [px_w(w, wpx) for w in [5,10,15,20]]
        hspace5, hspace10, hspace15, hspace20 = [px_h(h, hpx) for h in [5,10,15,20]]
        
        # create title widget
        self.title = QtWidgets.QPushButton()
        bh = px_h(40, hpx)
        self.title.setFixedHeight(bh)
        rad = int(bh/4)
        bw = px_h(3, hpx)
        # create gradient color for title button
        dkR, dkG, dkB = (85,233,255)
        mdR, mdG, mdB = (142,246,255)
        ltR, ltG, ltB = (184,254,255)
        self.title.setStyleSheet('QPushButton' 
                                 '{background-color : qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,'
                                                      f'stop:0    rgb({dkR},{dkG},{dkB}),'
                                                      f'stop:0.25 rgb({mdR},{mdG},{mdB}),'
                                                      f'stop:0.5  rgb({ltR},{ltG},{ltB}),'
                                                      f'stop:0.75 rgb({mdR},{mdG},{mdB}),'
                                                      f'stop:1    rgb({dkR},{dkG},{dkB}));'
                                 f'border : {bw}px outset gray;'
                                 f'border-radius : {rad};'
                                 '}')
        # create title text 
        txt_label = QtWidgets.QLabel('Keyboard Inputs')
        txt_label.setAlignment(QtCore.Qt.AlignCenter)
        txt_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        f = QtGui.QFont()
        f.setPointSize(14)
        f.setBold(True)
        txt_label.setFont(f)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(txt_label)
        self.title.setLayout(layout)
        
        # set up grid layout for displaying keyboard info
        self.keywidget = QtWidgets.QFrame()
        self.keywidget.setStyleSheet('QFrame {background-color : rgba(255,255,255,180)}')
        self.keygrid = QtWidgets.QGridLayout(self.keywidget)
        self.keygrid.setContentsMargins(wspace20,hspace20,wspace20,hspace20)
        
        # info for data viewing keys
        keyLayout1 = QtWidgets.QVBoxLayout()
        keyLayout1.setSpacing(hspace20)
        keylay1 = QtWidgets.QVBoxLayout()
        keylay1.setSpacing(hspace10)
        keytitle1 = QtWidgets.QLabel('Data Viewing')
        keytitle1.setAlignment(QtCore.Qt.AlignCenter)
        keytitle1.setFont(subheaderFont)
        keydict1 = {'e' : 'show EEG / switch EEG channel',
                    'm' : 'show EMG / switch EMG channel', 
                    'l' : 'show LFP / switch LFP channel',
                    't' : 'show / hide threshold for P-wave detection',
                    'p' : 'show / hide P-wave indices',
                    'o' : 'show / hide opotogenetic laser train',
                    'a' : 'show / hide EMG amplitude',
                    'f' : 'show / hide P-wave frequency',
                    'g' : 'show / hide GCaMP calcium signal',
                    'u' : 'show / hide signal underlying P-wave threshold',
                    'd' : 'show / hide standard deviation of LFP',
                    'b' : 'show / toggle through EEG freq. bands (<b>B</b> to hide)'}
        for key,txt in keydict1.items():
            row = self.key_def_row(key, txt, spacing=wspace20, uline=True, 
                                   keyFont=keyFont, txtFont=font)
            keylay1.addLayout(row)
        keyLayout1.addWidget(keytitle1, stretch=0)
        keyLayout1.addLayout(keylay1, stretch=2)
        
        # info for brain state keys
        keyLayout2 = QtWidgets.QVBoxLayout()
        keyLayout2.setSpacing(hspace20)
        keylay2 = QtWidgets.QVBoxLayout()
        keylay2.setSpacing(hspace10)
        keytitle2 = QtWidgets.QLabel('Brain State Annotation')
        keytitle2.setAlignment(QtCore.Qt.AlignCenter)
        keytitle2.setFont(subheaderFont)
        keydict2 = {'r' : 'REM sleep',
                    'w' : 'wake', 
                    'n' : 'non-REM sleep',
                    'i' : 'intermediate (transition) sleep',
                    'j' : 'failed transition sleep'}
        for key,txt in keydict2.items():
            row = self.key_def_row(key, txt, spacing=wspace20, uline=True, 
                                   keyFont=keyFont, txtFont=font)
            keylay2.addLayout(row)
        keyLayout2.addWidget(keytitle2, stretch=0)
        keyLayout2.addLayout(keylay2, stretch=2)
        
        # info for signals/event annotation keys
        keyLayout3 = QtWidgets.QVBoxLayout()
        keyLayout3.setSpacing(hspace20)
        keylay3 = QtWidgets.QVBoxLayout()
        keylay3.setSpacing(hspace10)
        keytitle3 = QtWidgets.QLabel('Signal / Event Annotation')
        keytitle3.setAlignment(QtCore.Qt.AlignCenter)
        keytitle3.setFont(subheaderFont)
        keydict3 = {'9' : 'annotate waveform as P-wave', 
                    '0' : 'eliminate waveform as P-wave',
                    'x' : 'annotate selected signal as noise',
                    'c' : 'annotate selected signal as clean'}
        for key,txt in keydict3.items():
            row = self.key_def_row(key, txt, spacing=wspace20, uline=False, 
                                   keyFont=keyFont, txtFont=font)
            keylay3.addLayout(row)
        keyLayout3.addWidget(keytitle3, stretch=0)
        keyLayout3.addLayout(keylay3, stretch=2)
        
        # info for plot adjustment keys
        keyLayout4 = QtWidgets.QVBoxLayout()
        keyLayout4.setSpacing(hspace20)
        keylay4 = QtWidgets.QGridLayout()
        keylay4.setVerticalSpacing(hspace10)
        keytitle4 = QtWidgets.QLabel('Plot Adjustment')
        keytitle4.setAlignment(QtCore.Qt.AlignCenter)
        keytitle4.setFont(subheaderFont)
        keydict4a = {'\u2190' : 'scroll to the left',
                     '\u2192' : 'scroll to the right',
                     '\u2193' : 'brighten EEG spectrogram',
                     '\u2191' : 'darken EEG spectrogram'}
        bsize, radius, bwidth = [px_w(w, wpx) for w in [30,8,3]]
        bfont = QtGui.QFont()
        bfont.setPointSize(15)
        bfont.setBold(True)
        bfont.setWeight(75)
        for i,(key,txt) in enumerate(keydict4a.items()):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(wspace20)
            # create icon for keyboard arrows
            btn = QtWidgets.QPushButton()
            btn.setFixedSize(bsize,bsize)
            btn.setStyleSheet('QPushButton'
                              '{ background-color : white;'
                              f'border : {bwidth}px solid black;'
                              f'border-radius : {radius};'
                              'padding : 1px }')
            btn.setText(key)
            btn.setFont(bfont)
            # create text label
            t = QtWidgets.QLabel(txt)
            t.setAlignment(QtCore.Qt.AlignLeft)
            t.setFont(font)
            keylay4.addWidget(btn, i, 0, alignment=QtCore.Qt.AlignCenter)
            keylay4.addWidget(t, i, 1, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignLeft)
        keydict4b = {'space' : 'collect sequence for annotation',
                     '1' : 'seconds time scale',
                     '2' : 'minutes time scale',
                     '3' : 'hours time scale',
                     's' : 'save brain state annotation'}
        for j,(key,txt) in enumerate(keydict4b.items()):
            row = self.key_def_row(key, txt, uline=False, keyFont=keyFont, txtFont=font)
            # collect QLabel objects from row, manually insert into grid layout
            keylay4.addWidget(row.itemAt(0).widget(), j+i+1, 0, 
                              alignment=QtCore.Qt.AlignCenter)
            keylay4.addWidget(row.itemAt(1).widget(), j+i+1, 1, 
                              alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignLeft)
        keyLayout4.addWidget(keytitle4, stretch=0)
        keyLayout4.addLayout(keylay4, stretch=2)
        # add layouts to main grid
        self.keygrid.addLayout(keyLayout1, 0, 0, 3, 1)
        self.keygrid.addWidget(vline(), 0, 1, 3, 1)
        self.keygrid.addLayout(keyLayout2, 0, 2, 1, 1)
        self.keygrid.addWidget(vline(orientation='h'), 1, 2, 1, 1)
        self.keygrid.addLayout(keyLayout3, 2, 2, 1, 1)
        self.keygrid.addWidget(vline(), 0, 3, 3, 1)
        self.keygrid.addLayout(keyLayout4, 0, 4, 3, 1)
        self.keygrid.setHorizontalSpacing(px_w(50, wpx))
        self.keygrid.setVerticalSpacing(px_h(30, hpx))
        self.centralLayout.addWidget(self.title)
        self.centralLayout.addWidget(self.keywidget)
        self.centralLayout.setSpacing(hspace20)
    
    
    def key_def_row(self, key, txt, spacing=10, uline=True, 
                    keyFont=QtGui.QFont(), txtFont=QtGui.QFont()):
        """
        Create informational text row
        @Params
        key - symbol on keyboard
        txt - text explanation of associated keypress action
        spacing - space between $key and $txt labels
        uline - if True, underline the first instance of $key in the $txt string
        keyFont, txtFont - fonts to use for key and text labels
        @Returns
        row - QHBoxLayout containing QLabel objects for key and text
        """
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(spacing)
        # create key label
        #k = QtWidgets.QLabel(key.upper())
        k = QtWidgets.QLabel(key)
        k.setAlignment(QtCore.Qt.AlignCenter)
        k.setFont(keyFont)
        # find and underline the first $key instance that begins a word in the $txt string
        if uline:
            ik = [i for i,l in enumerate(txt.lower()) if l==key.lower() and (i==0 or txt[i-1]==' ')]
            if len(ik) > 0:
                txt = txt[0:ik[0]] + '<u>' + txt[ik[0]] + '</u>' + txt[ik[0]+1 : ]
        # create text label
        t = QtWidgets.QLabel(txt)
        t.setAlignment(QtCore.Qt.AlignLeft)
        t.setFont(txtFont)
        row.addWidget(k, alignment=QtCore.Qt.AlignCenter, stretch=0)
        row.addWidget(t, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignLeft, stretch=2)
        return row


class GraphEEG(pg.PlotItem):
    """
    Custom graph allowing the user to select a segment of plotted data. 
    
    A selected sequence consists of the indices between two time points, which are 
    saved in the main window when the user double-clicks the corresponding point on 
    the plot. Once both the start and end points are established, the user can press
    the "X" key to annotate the selected sequence as "noise", or the "C" key to
    annotate the sequence as "clean". After the selected data is annotated, the
    bookend indices in the main window are cleared, and the next double-click
    within the plot boundaries is saved as the start index for a new sequence.
    """
    def __init__(self, parent, *args, **kwargs):
        """
        Instantiate plot object
        """
        super().__init__(*args, **kwargs)
        
        self.mainWin = parent
        # load ArrowButton objects from main window
        self.arrowStart_btn = self.mainWin.arrowStart_btn
        self.arrowEnd_btn = self.mainWin.arrowEnd_btn
        
        
    def mouseDoubleClickEvent(self, event):
        """
        Update start or end point of user-selected sequence
        """
        if event.button() == QtCore.Qt.LeftButton:
            # convert mouse click position to index in recording
            point = self.vb.mapSceneToView(event.pos()).x()
            point = np.round(point, np.abs(int(np.floor(np.log10(self.mainWin.dt)))))
            i = np.abs(self.mainWin.tseq - point).argmin()
            idx = int(self.mainWin.tseq[int(i)] * self.mainWin.sr)
            
            self.mainWin.show_arrowStart = True
            self.mainWin.show_arrowEnd = True
            # one or zero noise boundaries set
            if self.mainWin.noiseEndIdx is None:
                # no noise boundaries set - clicked point is starting noise idx
                if self.mainWin.noiseStartIdx is None:
                    self.mainWin.show_arrowEnd = False
                    self.mainWin.noiseStartIdx = idx
                    self.arrowStart_btn.active = False
                    self.arrowStart_btn.update_style()
                # clicked point is later in recording than starting noise idx
                elif self.mainWin.noiseStartIdx <= idx:
                    self.mainWin.noiseEndIdx = idx
                    self.arrowStart_btn.active = False
                    self.arrowStart_btn.update_style()
                    self.arrowEnd_btn.active = True
                    self.arrowEnd_btn.update_style()
                # clicked point is earlier in recording than starting noise idx
                elif self.mainWin.noiseStartIdx > idx:
                    self.mainWin.noiseEndIdx = int(self.mainWin.noiseStartIdx)
                    self.arrowEnd_btn.active = False
                    self.arrowEnd_btn.update_style()
                    self.mainWin.noiseStartIdx = idx
                    self.arrowStart_btn.active = True
                    self.arrowStart_btn.update_style()
            # both noise boundaries set
            else:
                # adjust starting noise idx
                if self.arrowStart_btn.active == True:
                    # clicked point is earlier in recording than ending noise idx
                    if self.mainWin.noiseEndIdx >= idx:
                        self.mainWin.noiseStartIdx = idx
                    # clicked point is later in recording than ending noise idx
                    else:
                        self.mainWin.noiseStartIdx = int(self.mainWin.noiseEndIdx)
                        self.arrowStart_btn.active = False
                        self.arrowStart_btn.update_style()
                        self.mainWin.noiseEndIdx = idx
                        self.arrowEnd_btn.active = True
                        self.arrowEnd_btn.update_style()
                # adjust ending noise idx
                elif self.arrowEnd_btn.active == True:
                    # clicked point is later in recording than starting noise idx
                    if self.mainWin.noiseStartIdx <= idx:
                        self.mainWin.noiseEndIdx = idx
                    # clicked point is earlier in recording than ending noise idx
                    else:
                        self.mainWin.noiseEndIdx = int(self.mainWin.noiseStartIdx)
                        self.arrowEnd_btn.active = False
                        self.arrowEnd_btn.update_style()
                        self.mainWin.noiseStartIdx = idx
                        self.arrowStart_btn.active = True
                        self.arrowStart_btn.update_style()
        # if user double-clicks with right button, clear selected points
        else:
            self.mainWin.noiseStartIdx = None
            self.arrowStart_btn.active = False
            self.mainWin.show_arrowStart = False
            self.mainWin.noiseEndIdx = None
            self.arrowEnd_btn.active = False
            self.mainWin.show_arrowEnd = False
        # update live plot in main window
        self.mainWin.plot_eeg(findPwaves=False, findArtifacts=False)
