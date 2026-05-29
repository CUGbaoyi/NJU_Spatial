import math
import tkinter as tk
from tkinter import ttk, messagebox

# Constants used by GCJ-02 transform.
_PI = math.pi
_A = 6378245.0
_EE = 0.00669342162296594323


def out_of_china(lat: float, lon: float) -> bool:
    """Return True if the coordinate is outside mainland China."""
    return lon < 72.004 or lon > 137.8347 or lat < 0.8293 or lat > 55.8271


def transform_lat(x: float, y: float) -> float:
    ret = (
        -100.0
        + 2.0 * x
        + 3.0 * y
        + 0.2 * y * y
        + 0.1 * x * y
        + 0.2 * math.sqrt(abs(x))
    )
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * _PI) + 40.0 * math.sin(y / 3.0 * _PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * _PI) + 320 * math.sin(y * _PI / 30.0)) * 2.0 / 3.0
    return ret


def transform_lon(x: float, y: float) -> float:
    ret = (
        300.0
        + x
        + 2.0 * y
        + 0.1 * x * x
        + 0.1 * x * y
        + 0.1 * math.sqrt(abs(x))
    )
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * _PI) + 40.0 * math.sin(x / 3.0 * _PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * _PI) + 300.0 * math.sin(x / 30.0 * _PI)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 to GCJ-02."""
    if out_of_china(lat, lon):
        return lat, lon

    d_lat = transform_lat(lon - 105.0, lat - 35.0)
    d_lon = transform_lon(lon - 105.0, lat - 35.0)
    rad_lat = lat / 180.0 * _PI
    magic = math.sin(rad_lat)
    magic = 1 - _EE * magic * magic
    sqrt_magic = math.sqrt(magic)

    d_lat = (d_lat * 180.0) / (((_A * (1 - _EE)) / (magic * sqrt_magic)) * _PI)
    d_lon = (d_lon * 180.0) / ((_A / sqrt_magic * math.cos(rad_lat)) * _PI)

    mg_lat = lat + d_lat
    mg_lon = lon + d_lon
    return mg_lat, mg_lon


def gcj02_to_wgs84(lat: float, lon: float) -> tuple[float, float]:
    """Convert GCJ-02 to WGS84 using an iterative inverse method."""
    if out_of_china(lat, lon):
        return lat, lon

    g_lat, g_lon = lat, lon
    w0_lat, w0_lon = g_lat, g_lon
    g1_lat, g1_lon = wgs84_to_gcj02(w0_lat, w0_lon)

    w1_lat = w0_lat - (g1_lat - g_lat)
    w1_lon = w0_lon - (g1_lon - g_lon)
    delta_lat = w1_lat - w0_lat
    delta_lon = w1_lon - w0_lon

    while abs(delta_lat) >= 1e-6 or abs(delta_lon) >= 1e-6:
        w0_lat, w0_lon = w1_lat, w1_lon
        g1_lat, g1_lon = wgs84_to_gcj02(w0_lat, w0_lon)
        w1_lat = w0_lat - (g1_lat - g_lat)
        w1_lon = w0_lon - (g1_lon - g_lon)
        delta_lat = w1_lat - w0_lat
        delta_lon = w1_lon - w0_lon

    return w1_lat, w1_lon


class CoordinateConverterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("WGS84 <-> GCJ-02 坐标转换")
        self.root.geometry("520x360")
        self.root.resizable(False, False)

        self.system_options = ["WGS84", "GCJ-02"]

        main = ttk.Frame(root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(main, text="坐标系转换器", font=("Microsoft YaHei UI", 15, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 12), sticky="w")

        ttk.Label(main, text="输入坐标系:").grid(row=1, column=0, sticky="w", pady=6)
        self.input_combo = ttk.Combobox(main, values=self.system_options, state="readonly", width=24)
        self.input_combo.current(0)
        self.input_combo.grid(row=1, column=1, sticky="w", pady=6)

        ttk.Label(main, text="输出坐标系:").grid(row=2, column=0, sticky="w", pady=6)
        self.output_combo = ttk.Combobox(main, values=self.system_options, state="readonly", width=24)
        self.output_combo.current(1)
        self.output_combo.grid(row=2, column=1, sticky="w", pady=6)

        ttk.Label(main, text="输入经度 (lon):").grid(row=3, column=0, sticky="w", pady=6)
        self.lon_entry = ttk.Entry(main, width=28)
        self.lon_entry.grid(row=3, column=1, sticky="w", pady=6)

        ttk.Label(main, text="输入纬度 (lat):").grid(row=4, column=0, sticky="w", pady=6)
        self.lat_entry = ttk.Entry(main, width=28)
        self.lat_entry.grid(row=4, column=1, sticky="w", pady=6)

        convert_btn = ttk.Button(main, text="转换", command=self.convert)
        convert_btn.grid(row=5, column=0, columnspan=2, pady=(12, 12))

        ttk.Separator(main, orient="horizontal").grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.result_title = ttk.Label(main, text="转换结果", font=("Microsoft YaHei UI", 11, "bold"))
        self.result_title.grid(row=7, column=0, columnspan=2, sticky="w")

        self.result_lon = ttk.Label(main, text="经度: -")
        self.result_lon.grid(row=8, column=0, columnspan=2, sticky="w", pady=4)

        self.result_lat = ttk.Label(main, text="纬度: -")
        self.result_lat.grid(row=9, column=0, columnspan=2, sticky="w", pady=4)

        hint = "提示: 若坐标不在中国大陆范围，程序会返回原坐标。"
        ttk.Label(main, text=hint, foreground="#555555").grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(14, 0)
        )

    @staticmethod
    def normalize_system_name(name: str) -> str:
        if "WGS84" in name:
            return "WGS84"
        return "GCJ-02"

    def convert(self) -> None:
        input_sys = self.normalize_system_name(self.input_combo.get())
        output_sys = self.normalize_system_name(self.output_combo.get())

        try:
            lon = float(self.lon_entry.get().strip())
            lat = float(self.lat_entry.get().strip())
        except ValueError:
            messagebox.showerror("输入错误", "经度和纬度必须是数字。")
            return

        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            messagebox.showerror("输入错误", "请输入有效范围内的经纬度。")
            return

        if input_sys == output_sys:
            new_lat, new_lon = lat, lon
        elif input_sys == "WGS84" and output_sys == "GCJ-02":
            new_lat, new_lon = wgs84_to_gcj02(lat, lon)
        elif input_sys == "GCJ-02" and output_sys == "WGS84":
            new_lat, new_lon = gcj02_to_wgs84(lat, lon)
        else:
            messagebox.showerror("转换错误", "不支持的坐标系类型。")
            return

        self.result_lon.config(text=f"经度: {new_lon:.8f}")
        self.result_lat.config(text=f"纬度: {new_lat:.8f}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CoordinateConverterApp(root)
    root.mainloop()
