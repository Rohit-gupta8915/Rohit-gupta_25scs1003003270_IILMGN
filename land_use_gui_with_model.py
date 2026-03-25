# land_use_gui_with_model.py
# Tkinter GUI that integrates the newly trained ML model
# - Expects model file at:
#   C:\Users\Rohit Gupta\PYTHON\data\landuse_pipeline_gui.pkl
#
# Run: python land_use_gui_with_model.py

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# ---------------- Config: model path ----------------
MODEL_PATH = Path(r"C:\Users\Rohit Gupta\PYTHON\data\landuse_pipeline_gui.pkl")

# ---------------- Rule-based recommendation (unchanged) ----------------
def recommendation_rule(population, soil, area_type, lat, lon):
    rec = ""
    reasons = []

    at = area_type.lower()
    if at == "forest":
        rec = "Conserve / Reforest"
        reasons.append("Area classified as forest → conservation priority.")
        if soil < 4:
            reasons.append("Low soil fertility → focus on soil restoration.")
        else:
            reasons.append("Soil acceptable → protect and consider eco-tourism.")
    elif at == "agriculture":
        if soil >= 7:
            rec = "Intensive Sustainable Agriculture"
            reasons.append("High soil fertility → suitable for crops with sustainable practices.")
            if population > 20000:
                reasons.append("High population nearby → market-oriented farming.")
        else:
            rec = "Soil improvement + Agroforestry"
            reasons.append("Low/medium soil fertility → improve soil and consider agroforestry.")
    elif at == "urban":
        if population > 50000:
            rec = "Urban Development (High Priority) with Green Infrastructure"
            reasons.append("High population → suitable for development with green planning.")
        else:
            rec = "Planned Low-Density Development"
            reasons.append("Lower population → planned development to avoid sprawl.")
    else:
        rec = "Mixed-Use Planning"
        reasons.append("Mixed land type → combine conservation, agriculture, and controlled development.")
        if soil >= 7:
            reasons.append("Good soil in patches → allocate to sustainable agriculture.")
        if soil < 4:
            reasons.append("Poor soil in patches → consider reforestation or restoration.")

    if soil <= 3:
        reasons.append("Soil fertility low → conduct soil tests, use organic amendments.")
    elif soil >= 8:
        reasons.append("Soil fertility high → maintain with crop rotation and organic matter.")

    return rec, reasons

# ---------------- Model loader helper ----------------
def load_model(path: Path):
    """
    Returns (model_pipeline, label_map, path_str) or (None, None, None) if not found/failed.
    The training script saved a dict: {"model": model_pipeline, "label_map": label_map}
    """
    if not path.exists():
        return None, None, None
    try:
        data = joblib.load(path)
    except Exception as e:
        print("Failed to load model file:", e)
        return None, None, None

    # If file is a dict with keys 'model' and 'label_map'
    if isinstance(data, dict) and "model" in data and "label_map" in data:
        return data["model"], data["label_map"], str(path)
    # If file is pipeline directly (older format)
    if hasattr(data, "predict"):
        # no label_map saved — hard to map codes to labels; return pipeline only
        return data, None, str(path)
    # unknown structure
    return None, None, None

# ---------------- Build DataFrame row for model input ----------------
def build_input_row(population, soil, area_type, lat, lon):
    """
    Build a DataFrame with columns exactly:
    ['population','soil_fertility','area_type','latitude','longitude']
    This matches the training script.
    """
    row = {
        "population": float(population),
        "soil_fertility": int(soil),
        "area_type": str(area_type),
        "latitude": float(lat),
        "longitude": float(lon)
    }
    return pd.DataFrame([row], columns=["population","soil_fertility","area_type","latitude","longitude"])

# ---------------- Load model at startup ----------------
model_pipeline, label_map, model_used_path = load_model(MODEL_PATH)
MODEL_AVAILABLE = model_pipeline is not None

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Land-Use Advisor (ML-enabled)")
root.geometry("760x620")
root.resizable(False, False)

title = tk.Label(root, text="Land-Use Recommendation System (ML & Rules)", font=("Arial", 16, "bold"))
title.pack(pady=8)

frame = tk.Frame(root)
frame.pack(pady=6)

# Population
tk.Label(frame, text="Nearby Population:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
pop_entry = tk.Entry(frame); pop_entry.grid(row=0, column=1, padx=8, pady=6)
pop_entry.insert(0, "10000")

# Soil Fertility
tk.Label(frame, text="Soil Fertility (0–10):").grid(row=1, column=0, sticky="w", padx=8, pady=6)
soil_scale = tk.Scale(frame, from_=0, to=10, orient=tk.HORIZONTAL, length=240)
soil_scale.set(5); soil_scale.grid(row=1, column=1, padx=8, pady=6)

# Area Type
tk.Label(frame, text="Area Type:").grid(row=2, column=0, sticky="w", padx=8, pady=6)
area_var = tk.StringVar()
area_dropdown = ttk.Combobox(frame, textvariable=area_var, values=["Agriculture","Urban","Forest","Mixed"], state="readonly")
area_dropdown.current(0); area_dropdown.grid(row=2, column=1, padx=8, pady=6)

# Latitude & Longitude
tk.Label(frame, text="Latitude:").grid(row=3, column=0, sticky="w", padx=8, pady=6)
lat_entry = tk.Entry(frame); lat_entry.grid(row=3, column=1, padx=8, pady=6)
lat_entry.insert(0, "22.0")

tk.Label(frame, text="Longitude:").grid(row=4, column=0, sticky="w", padx=8, pady=6)
lon_entry = tk.Entry(frame); lon_entry.grid(row=4, column=1, padx=8, pady=6)
lon_entry.insert(0, "79.0")

# ML use checkbox
use_model_var = tk.BooleanVar(value=MODEL_AVAILABLE)
if MODEL_AVAILABLE:
    model_label_text = f"Use trained ML model (found: {model_used_path})"
else:
    model_label_text = "Use trained ML model (NOT found)"
model_chk = tk.Checkbutton(root, text=model_label_text, variable=use_model_var)
model_chk.pack(pady=6)

# Result box
result_box = tk.Text(root, width=92, height=18, state="disabled", wrap="word")
result_box.pack(padx=10, pady=8)

# Helper to format and show results
def show_result(text):
    result_box.config(state="normal")
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, text)
    result_box.config(state="disabled")

# Action on click
def on_generate():
    # validate inputs
    try:
        population = float(pop_entry.get())
        soil = int(soil_scale.get())
        area_type = area_var.get()
        lat = float(lat_entry.get())
        lon = float(lon_entry.get())
    except Exception:
        messagebox.showerror("Input Error", "Please enter valid numeric values for population/latitude/longitude.")
        return

    # rule-based always
    rec, reasons = recommendation_rule(population, soil, area_type, lat, lon)

    ml_text = ""
    if use_model_var.get():
        if not MODEL_AVAILABLE:
            ml_text = "ML model not available on this machine.\n\n"
        else:
            try:
                df_row = build_input_row(population, soil, area_type, lat, lon)
                pred = model_pipeline.predict(df_row)
                # pred may be numeric code; map via label_map if present
                if label_map is not None:
                    # pred may be array like [0], ensure int(key)
                    pred_code = int(pred[0])
                    pred_label = label_map.get(pred_code, str(pred_code))
                else:
                    # if no label_map, assume classifier returned string label
                    pred_label = str(pred[0])
                ml_text = f"ML model prediction: {pred_label}\n"
                # prob if available
                try:
                    if hasattr(model_pipeline, "predict_proba"):
                        proba = model_pipeline.predict_proba(df_row)[0]
                        # find max prob
                        max_idx = int(np.argmax(proba))
                        max_prob = float(np.max(proba))
                        if label_map is not None:
                            prob_label = label_map.get(max_idx, str(max_idx))
                        else:
                            prob_label = str(max_idx)
                        ml_text += f"Confidence: {max_prob:.2f} for {prob_label}\n"
                except Exception:
                    pass
                ml_text += "\n"
            except Exception as e:
                ml_text = f"ML model could not be applied: {e}\n\n"

    # aggregate display
    out = ""
    if ml_text:
        out += ml_text
    out += "Rule-based recommendation:\n"
    out += f"👉 {rec}\n\nReasons:\n"
    for r in reasons:
        out += f"• {r}\n"

    show_result(out)

# Button
btn = tk.Button(root, text="Generate Recommendation", command=on_generate, bg="#2e8b57", fg="white", font=("Arial", 12))
btn.pack(pady=6)

# Note
note = tk.Label(root, text="Note: ML model learned the rule-based recommendations from training data. If the predicted label does not match intuition, use the rule-based recommendation shown below.", fg="gray")
note.pack(pady=4)

root.mainloop()
