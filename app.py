import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "data/ford_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None

# Beispiel-Testdaten (ersetze durch echte Testdaten für Gütekriterien)
X_test = pd.DataFrame(
    {
        "model": ["Fiesta", "Focus", "Fiesta", "Focus", "Fiesta"],
        "year": [2015, 2016, 2017, 2018, 2019],
        "transmission": ["Manual", "Manual", "Automatic", "Manual", "Automatic"],
        "mileage": [50000, 40000, 30000, 20000, 10000],
        "fuelType": ["Petrol", "Diesel", "Petrol", "Diesel", "Petrol"],
        "tax": [30, 20, 30, 20, 30],
        "mpg": [55.4, 60.1, 58.0, 61.4, 59.7],
        "engineSize": [1.0, 1.5, 1.0, 1.5, 1.0],
    }
)
y_test = np.array([12000, 13000, 14000, 15000, 16000])


def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import mean_squared_error, r2_score

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2


def schätzen():
    if model is None:
        messagebox.showerror("Fehler", "Kein Modell geladen!")
        return

    try:
        input_data = {
            "model": combo_model.get(),
            "year": int(entry_year.get()),
            "transmission": combo_transmission.get(),
            "mileage": int(entry_mileage.get()),
            "fuelType": combo_fuel.get(),
            "tax": float(entry_tax.get()),
            "mpg": float(entry_mpg.get()),
            "engineSize": float(entry_engine.get()),
        }
    except Exception as e:
        messagebox.showerror("Fehler", f"Bitte alle Felder korrekt ausfüllen!\n{e}")
        return

    df = pd.DataFrame([input_data])

    try:
        preis = model.predict(df)[0]
    except Exception as e:
        messagebox.showerror("Fehler", f"Vorhersage fehlgeschlagen: {e}")
        return

    try:
        rmse, r2 = evaluate_model(model, X_test, y_test)
    except Exception:
        rmse, r2 = None, None

    result = f"Geschätzter Preis: {preis:,.2f} €"
    if rmse is not None and r2 is not None:
        result += f"\nRMSE: {rmse:,.2f}\nR²: {r2:.3f}"
    else:
        result += "\n(Gütekriterien konnten nicht berechnet werden)"

    label_result.config(text=result)


root = tk.Tk()
root.title("Auto-Preis Schätzung")

# Beispielwerte für Dropdowns (ggf. an dein Modell anpassen)
modelle = ["Fiesta", "Focus", "EcoSport", "Grand C-MAX", "Kuga", "Cougar"]
transmissions = ["Manual", "Automatic", "Semi-Auto"]
fuel_types = ["Petrol", "Diesel", "Hybrid", "Electric"]

fields = [
    ("Modell:", "model", ttk.Combobox, {"values": modelle}),
    ("Baujahr:", "year", tk.Entry, {}),
    ("Getriebe:", "transmission", ttk.Combobox, {"values": transmissions}),
    ("Kilometerstand:", "mileage", tk.Entry, {}),
    ("Kraftstoff:", "fuelType", ttk.Combobox, {"values": fuel_types}),
    ("Steuer (€):", "tax", tk.Entry, {}),
    ("mpg:", "mpg", tk.Entry, {}),
    ("Motorgröße:", "engineSize", tk.Entry, {}),
]

widgets = {}
for i, (label_text, key, widget_class, options) in enumerate(fields):
    tk.Label(root, text=label_text).grid(row=i, column=0, padx=10, pady=5, sticky="e")
    if widget_class == ttk.Combobox:
        widget = widget_class(root, state="readonly", **options)
        widget.current(0)
    else:
        widget = widget_class(root)
    widget.grid(row=i, column=1, padx=10, pady=5)
    widgets[key] = widget

combo_model = widgets["model"]
entry_year = widgets["year"]
combo_transmission = widgets["transmission"]
entry_mileage = widgets["mileage"]
combo_fuel = widgets["fuelType"]
entry_tax = widgets["tax"]
entry_mpg = widgets["mpg"]
entry_engine = widgets["engineSize"]

button = tk.Button(root, text="Preis schätzen", command=schätzen)
button.grid(row=len(fields), column=0, columnspan=2, pady=10)

label_result = tk.Label(root, text="", fg="blue", font=("Arial", 12))
label_result.grid(row=len(fields) + 1, column=0, columnspan=2, pady=10)

root.mainloop()
