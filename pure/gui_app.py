#!/usr/bin/env python3
"""
GUI Application for Pure Panorama Stitching
Sử dụng tkinter để tạo giao diện đơn giản và thân thiện.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
from pathlib import Path

# Import pure implementation
try:
    from .image_io import read_images, write_image
    from .panorama_stitcher import PanoramaStitcher
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pure.image_io import read_images, write_image
    from pure.panorama_stitcher import PanoramaStitcher


class PanoramaGUI:
    """GUI Application cho panorama stitching."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pure Panorama Stitcher")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.image_paths = []
        self.panorama_result = None
        self.is_processing = False
        
        # Setup UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup giao diện người dùng."""
        
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🔬 Pure Panorama Stitcher",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(
            header_frame,
            text="No OpenCV - Pure Python Implementation",
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='#ffffff', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.config(width=300)
        
        # Control section
        control_label = tk.Label(
            left_panel,
            text="Controls",
            font=('Arial', 14, 'bold'),
            bg='#ffffff',
            fg='#2c3e50',
            anchor='w'
        )
        control_label.pack(fill=tk.X, padx=15, pady=(15, 10))
        
        # Select images button
        select_btn = tk.Button(
            left_panel,
            text="📁 Chọn Ảnh",
            font=('Arial', 12),
            bg='#3498db',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2',
            command=self._select_images
        )
        select_btn.pack(fill=tk.X, padx=15, pady=5)
        
        # Selected images list
        list_label = tk.Label(
            left_panel,
            text="Ảnh đã chọn:",
            font=('Arial', 10, 'bold'),
            bg='#ffffff',
            fg='#2c3e50',
            anchor='w'
        )
        list_label.pack(fill=tk.X, padx=15, pady=(15, 5))
        
        # Listbox with scrollbar
        list_frame = tk.Frame(left_panel, bg='#ffffff')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox = tk.Listbox(
            list_frame,
            font=('Arial', 9),
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            bg='#ffffff',
            fg='#2c3e50',
            selectbackground='#3498db',
            selectforeground='white'
        )
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        
        # Remove selected button
        remove_btn = tk.Button(
            left_panel,
            text="🗑️ Xóa Ảnh Đã Chọn",
            font=('Arial', 10),
            bg='#e74c3c',
            fg='white',
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor='hand2',
            command=self._remove_selected_image
        )
        remove_btn.pack(fill=tk.X, padx=15, pady=5)
        
        # Clear all button
        clear_btn = tk.Button(
            left_panel,
            text="🧹 Xóa Tất Cả",
            font=('Arial', 10),
            bg='#95a5a6',
            fg='white',
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor='hand2',
            command=self._clear_all_images
        )
        clear_btn.pack(fill=tk.X, padx=15, pady=5)
        
        # Separator
        separator = tk.Frame(left_panel, height=2, bg='#bdc3c7')
        separator.pack(fill=tk.X, padx=15, pady=15)
        
        # Settings section
        settings_label = tk.Label(
            left_panel,
            text="Settings",
            font=('Arial', 14, 'bold'),
            bg='#ffffff',
            fg='#2c3e50',
            anchor='w'
        )
        settings_label.pack(fill=tk.X, padx=15, pady=(10, 10))
        
        # Smoothing parameter
        smoothing_frame = tk.Frame(left_panel, bg='#ffffff')
        smoothing_frame.pack(fill=tk.X, padx=15, pady=5)
        
        tk.Label(
            smoothing_frame,
            text="Smoothing:",
            font=('Arial', 9),
            bg='#ffffff',
            fg='#2c3e50'
        ).pack(anchor='w')
        
        self.smoothing_var = tk.DoubleVar(value=0.10)
        smoothing_scale = tk.Scale(
            smoothing_frame,
            from_=0.05,
            to=0.30,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.smoothing_var,
            bg='#ffffff',
            fg='#2c3e50',
            troughcolor='#ecf0f1',
            activebackground='#3498db'
        )
        smoothing_scale.pack(fill=tk.X)
        
        # Stitch button
        self.stitch_btn = tk.Button(
            left_panel,
            text="🔧 Ghép Ảnh",
            font=('Arial', 14, 'bold'),
            bg='#27ae60',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=15,
            cursor='hand2',
            command=self._start_stitching
        )
        self.stitch_btn.pack(fill=tk.X, padx=15, pady=(20, 10))
        
        # Save button
        self.save_btn = tk.Button(
            left_panel,
            text="💾 Lưu Kết Quả",
            font=('Arial', 12),
            bg='#9b59b6',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2',
            command=self._save_result,
            state=tk.DISABLED
        )
        self.save_btn.pack(fill=tk.X, padx=15, pady=5)
        
        # Right panel - Preview & Result
        right_panel = tk.Frame(main_frame, bg='#ffffff', relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Preview/Result label
        self.result_label = tk.Label(
            right_panel,
            text="Chọn ảnh và nhấn 'Ghép Ảnh' để bắt đầu",
            font=('Arial', 12),
            bg='#ffffff',
            fg='#34495e'
        )
        self.result_label.pack(pady=20)
        
        # Canvas for image display
        canvas_frame = tk.Frame(right_panel, bg='#ffffff')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='#ecf0f1',
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Sẵn sàng",
            font=('Arial', 9),
            bg='#34495e',
            fg='white',
            anchor='w',
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _select_images(self):
        """Chọn nhiều ảnh từ file dialog."""
        filetypes = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        paths = filedialog.askopenfilenames(
            title="Chọn ảnh để ghép (theo thứ tự từ trái sang phải)",
            filetypes=filetypes
        )
        
        if paths:
            self.image_paths.extend(paths)
            self._update_image_list()
            self._update_status(f"Đã chọn {len(self.image_paths)} ảnh")
    
    def _update_image_list(self):
        """Cập nhật danh sách ảnh trong listbox."""
        self.image_listbox.delete(0, tk.END)
        for i, path in enumerate(self.image_paths, 1):
            filename = os.path.basename(path)
            self.image_listbox.insert(tk.END, f"{i}. {filename}")
    
    def _remove_selected_image(self):
        """Xóa ảnh đã chọn khỏi danh sách."""
        selection = self.image_listbox.curselection()
        if selection:
            index = selection[0]
            self.image_paths.pop(index)
            self._update_image_list()
            self._update_status(f"Còn lại {len(self.image_paths)} ảnh")
    
    def _clear_all_images(self):
        """Xóa tất cả ảnh."""
        if self.image_paths:
            if messagebox.askyesno("Xác nhận", "Xóa tất cả ảnh đã chọn?"):
                self.image_paths.clear()
                self._update_image_list()
                self.panorama_result = None
                self._clear_canvas()
                self.save_btn.config(state=tk.DISABLED)
                self._update_status("Đã xóa tất cả ảnh")
    
    def _start_stitching(self):
        """Bắt đầu quá trình ghép ảnh trong thread riêng."""
        if len(self.image_paths) < 2:
            messagebox.showwarning(
                "Cảnh báo",
                "Vui lòng chọn ít nhất 2 ảnh để ghép!"
            )
            return
        
        if self.is_processing:
            messagebox.showinfo("Thông báo", "Đang xử lý, vui lòng đợi...")
            return
        
        # Disable buttons
        self.stitch_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.is_processing = True
        
        # Start in separate thread
        thread = threading.Thread(target=self._stitch_images, daemon=True)
        thread.start()
    
    def _stitch_images(self):
        """Ghép ảnh (chạy trong thread riêng)."""
        try:
            self._update_status("Đang đọc ảnh...")
            
            # Read images
            images = read_images(self.image_paths)
            
            self._update_status(f"Đã đọc {len(images)} ảnh. Đang khởi tạo SIFT...")
            
            # Create stitcher
            stitcher = PanoramaStitcher(
                sift_params={
                    'num_octaves': 4,
                    'num_scales': 5,
                    'contrast_threshold': 0.01,
                    'edge_threshold': 20,
                    'border_width': 3,
                },
                matcher_params={
                    'ratio_threshold': 0.8,
                    'cross_check': True,
                },
                ransac_params={
                    'ransac_reproj_threshold': 4.0,
                    'max_iters': 3000,
                    'min_inliers': 8,
                },
                blending_params={
                    'smoothing_window_percent': self.smoothing_var.get(),
                }
            )
            
            self._update_status("Đang ghép ảnh... (Quá trình này có thể mất vài phút)")
            
            # Stitch
            result = stitcher.stitch_multiple(images)
            
            self._update_status("Hoàn thành! Đang hiển thị kết quả...")
            
            # Store result
            self.panorama_result = result
            
            # Display result
            self.root.after(0, self._display_result, result)
            
            self._update_status(f"✓ Hoàn thành! Panorama size: {result.shape}")
            
        except Exception as e:
            error_msg = f"Lỗi: {str(e)}"
            self._update_status(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Lỗi", error_msg))
        
        finally:
            # Re-enable buttons
            self.is_processing = False
            self.root.after(0, lambda: self.stitch_btn.config(state=tk.NORMAL))
            if self.panorama_result is not None:
                self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
    
    def _display_result(self, image):
        """Hiển thị kết quả trên canvas."""
        self._clear_canvas()
        
        # Convert numpy array to PIL Image
        if len(image.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode='L')
        else:
            # RGB
            pil_image = Image.fromarray(image, mode='RGB')
        
        # Resize if too large
        max_size = 1000
        if pil_image.width > max_size or pil_image.height > max_size:
            ratio = min(max_size / pil_image.width, max_size / pil_image.height)
            new_width = int(pil_image.width * ratio)
            new_height = int(pil_image.height * ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Display on canvas
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.photo,
            anchor=tk.CENTER
        )
        
        # Update scroll region
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
        self.result_label.config(
            text=f"Panorama ({pil_image.width} × {pil_image.height} pixels)",
            fg='#27ae60',
            bg='#ffffff'
        )
    
    def _clear_canvas(self):
        """Xóa canvas."""
        self.canvas.delete("all")
        self.result_label.config(
            text="Chọn ảnh và nhấn 'Ghép Ảnh' để bắt đầu",
            fg='#34495e',
            bg='#ffffff'
        )
    
    def _save_result(self):
        """Lưu kết quả ra file."""
        if self.panorama_result is None:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để lưu!")
            return
        
        filetypes = [
            ('JPEG files', '*.jpg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Lưu panorama",
            defaultextension='.jpg',
            filetypes=filetypes,
            initialdir=os.path.join(os.getcwd(), 'pure_outputs')
        )
        
        if filename:
            try:
                write_image(filename, self.panorama_result)
                messagebox.showinfo("Thành công", f"Đã lưu panorama tại:\n{filename}")
                self._update_status(f"Đã lưu: {filename}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu file:\n{str(e)}")
    
    def _update_status(self, message):
        """Cập nhật status bar."""
        self.root.after(0, lambda: self.status_bar.config(text=message))


def main():
    """Main function để chạy GUI."""
    root = tk.Tk()
    app = PanoramaGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

