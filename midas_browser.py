from PyQt6.QtCore import QUrl, QTimer, QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QIcon
import os
import sys
import subprocess
import time
import requests
import base64
import psutil

# Load the xenospinner.png and splash.jpeg images
with open('xenospinner.png', 'rb') as spinner_file, open('splashnew2.jpg', 'rb') as splash_file:
    spinner_data = spinner_file.read()
    splash_data = splash_file.read()

LOADING_HTML = f'''
<html>
<head>
    <style>
        @keyframes hover {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-10px); }}
        }}
        @keyframes zoom {{
            0%, 100% {{ background-size: 100% 100%; }}
            25% {{ background-size: 105% 105%; }}
            50% {{ background-size: 110% 110%; }}
            75% {{ background-size: 105% 105%; }}
        }}
        body {{
            background-image: url('data:image/jpeg;base64,{base64.b64encode(splash_data).decode("utf-8")}');
            background-size: cover;
            background-position: center;
            margin: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            align-items: center;
            position: relative;
            overflow: hidden;
            animation: zoom 20s ease-in-out infinite;
        }}
        body::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(ellipse at center, rgba(0,0,0,0) 0%, rgba(0,0,0,0.5) 100%);
            pointer-events: none;
        }}
        .content {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            flex-grow: 1;
        }}
        .spinner {{
            width: 120px;
            height: auto;
            animation: hover 2s ease-in-out infinite;
            position: relative;
            z-index: 1;
        }}
        .loading-text {{
            margin-top: 20px;
            font-family: Arial, sans-serif;
            font-size: 40px;
            color: #fff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            position: relative;
            z-index: 1;
        }}
        .footer {{
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #fff;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            padding: 10px;
            position: relative;
            z-index: 1;
        }}
    </style>
</head>
<body>
    <div class="content">
        <img src="data:image/png;base64,{base64.b64encode(spinner_data).decode("utf-8")}" class="spinner" alt="Loading...">
        <div class="loading-text">Starting MIDAS 2.0...</div>
    </div>
    <div class="footer">Developed by Xenovative.Ltd, 2025</div>
</body>
</html>
'''

class InstallDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('MIDAS First-Time Setup')
        self.setMinimumWidth(400)
        layout = QVBoxLayout()
        
        label = QLabel('This appears to be your first time running MIDAS. '
                       'We will now set up the necessary virtual environments '
                       'and download required dependencies.')
        label.setWordWrap(True)
        layout.addWidget(label)
        
        self.status_label = QLabel('Preparing installation...')
        layout.addWidget(self.status_label)
        
        install_button = QPushButton('Start Installation')
        install_button.clicked.connect(self.start_installation)
        layout.addWidget(install_button)
        
        self.setLayout(layout)
        
    def start_installation(self):
        try:
            self.status_label.setText('Installing virtual environments...')
            QApplication.processEvents()
            
            # Create a PowerShell process with proper execution policy
            powershell_command = [
                'powershell.exe',
                '-NoProfile',
                '-ExecutionPolicy', 'Bypass',
                '-Command', f'& "{os.path.join(os.getcwd(), "install_and_run.bat")}"'
            ]
            
            install_process = subprocess.Popen(
                powershell_command,
                cwd=os.path.dirname(__file__),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            # Wait for installation to complete
            stdout, stderr = install_process.communicate()
            
            if install_process.returncode == 0:
                self.status_label.setText('Installation successful!')
                QTimer.singleShot(1000, self.accept)
            else:
                self.status_label.setText(f'Installation failed: {stderr}')
                QMessageBox.critical(self, 'Installation Error', 
                                   f'Installation failed:\n{stderr}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

class ServiceThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def run(self):
        try:
            if check_first_time_run():
                # First time run - execute installation script
                install_process = subprocess.Popen([
                    'powershell.exe',
                    '-NoProfile',
                    '-ExecutionPolicy', 'Bypass',
                    '-Command', f'& "{os.path.join(os.getcwd(), "install_and_run.bat")}"'
                ], creationflags=subprocess.CREATE_NO_WINDOW)
                install_process.wait()
            else:
                # Regular run - start services directly
                service_process = subprocess.Popen([
                    'powershell.exe',
                    '-NoProfile',
                    '-ExecutionPolicy', 'Bypass',
                    '-Command', f'& "{os.path.join(os.getcwd(), "start_services.ps1")}"'
                ], creationflags=subprocess.CREATE_NO_WINDOW)
                service_process.wait()
            self.finished.emit()
        except subprocess.CalledProcessError as e:
            self.error.emit(f'Failed to start services: {e}')
        except Exception as e:
            self.error.emit(f'Unexpected error: {e}')

class MIDASBrowser(QWebEngineView):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MIDAS Browser')
        screen = QApplication.primaryScreen()
        size = screen.size()
        self.resize(size.width(), size.height())
        
        # Set the application icon
        icon_path = os.path.join(os.path.dirname(__file__), 'xenoapp.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Make the browser window always on top
        from PyQt6.QtCore import Qt
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        
        # Show loading screen initially
        self.setHtml(LOADING_HTML)
        
        # Setup the page to inject custom CSS for scrollbars
        self.loadFinished.connect(self.on_load_finished)
        
        # Store the scrollbar CSS to inject
        self.scrollbar_css = """
        /* Minimalistic and Round Scrollbar Styling - Global Styles */
        * {
            scrollbar-width: thin !important;  /* For Firefox */
            scrollbar-color: rgba(128, 128, 128, 0.5) transparent !important;  /* For Firefox */
        }

        /* Webkit (Chrome, Safari, newer versions of Opera) */
        *::-webkit-scrollbar {
            width: 8px !important;  /* Thin scrollbar */
            height: 8px !important;  /* Horizontal scrollbar */
            display: block !important;
        }

        *::-webkit-scrollbar-track {
            background: transparent !important;  /* Transparent track */
            border-radius: 10px !important;
        }

        *::-webkit-scrollbar-thumb {
            background-color: rgba(128, 128, 128, 0.5) !important;  /* Semi-transparent gray */
            border-radius: 10px !important;  /* Fully rounded scrollbar */
            border: 2px solid transparent !important;  /* Creates a slight padding effect */
            background-clip: content-box !important;  /* Ensures border doesn't affect size */
        }

        *::-webkit-scrollbar-thumb:hover {
            background-color: rgba(128, 128, 128, 0.7) !important;  /* Slightly darker on hover */
        }
        """

    def load_app(self):
        # Services are ready, load the application
        self.load(QUrl('http://127.0.0.1:7860'))
        
    def on_load_finished(self, ok):
        if ok:
            # Inject custom CSS for scrollbars
            js = f"""
            (function() {{
                const style = document.createElement('style');
                style.textContent = `{self.scrollbar_css}`;
                document.head.appendChild(style);
                console.log('MIDAS Browser: Custom scrollbar styling applied');
            }})();
            """
            self.page().runJavaScript(js)

def check_first_time_run():
    """Check if this is the first time running the application"""
    required_dirs = ['comfyUI', 'venvs']
    return not all(os.path.exists(os.path.join(os.getcwd(), dir)) for dir in required_dirs)

def main():
    app = QApplication(sys.argv)
    
    # Check if this is first-time setup
    if check_first_time_run():
        install_dialog = InstallDialog()
        if install_dialog.exec() != QDialog.DialogCode.Accepted:
            sys.exit(1)
    
    browser = MIDASBrowser()
    browser.show()
    
    # Start services in a separate thread
    service_thread = ServiceThread()
    service_thread.finished.connect(browser.load_app)
    service_thread.error.connect(lambda message: QMessageBox.critical(browser, 'Error', message))
    service_thread.start()

    # Register cleanup function to kill all processes
    def cleanup():
        # Stop the MIDAS app.py, comfyUI's main.py, and PowerShell processes
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] in ['python.exe', 'pythonw.exe', 'powershell.exe']:
                try:
                    cmdline = proc.cmdline()
                    if 'app.py' in cmdline or 'main.py' in cmdline or 'start_services.ps1' in cmdline:
                        proc.terminate()
                        try:
                            proc.wait(timeout=1)  # Wait for process to terminate
                        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        # Force kill any remaining processes
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] in ['python.exe', 'pythonw.exe', 'powershell.exe']:
                try:
                    cmdline = proc.cmdline()
                    if 'app.py' in cmdline or 'main.py' in cmdline or 'start_services.ps1' in cmdline:
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    
    app.aboutToQuit.connect(cleanup)
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
