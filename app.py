import pickle
import pandas as pd
import gradio as gr
import os

with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("cleaned_car_data.csv")

COMPANIES = sorted(df["company"].unique().tolist())
FUEL_TYPES = sorted(df["fuel_type"].unique().tolist())
NAMES_BY_COMPANY = (
    df.groupby("company")["name"]
    .apply(lambda x: sorted(x.unique().tolist()))
    .to_dict()
)
YEAR_MIN, YEAR_MAX = int(df["year"].min()), int(df["year"].max())
KMS_MAX = int(df["kms_driven"].max())


def update_names(company):
    names = NAMES_BY_COMPANY.get(company, [])
    return gr.update(choices=names, value=names[0] if names else None)


def predict_price(company, name, year, kms_driven, fuel_type):
    if not company or not name:
        return "Please select a company and car model."
    sample = pd.DataFrame([{
        "name": name,
        "company": company,
        "year": int(year),
        "kms_driven": int(kms_driven),
        "fuel_type": fuel_type,
    }])
    predicted = model.predict(sample)[0]
    predicted = max(predicted, 0)
    return f"₹ {predicted:,.0f}"


with gr.Blocks(title="Used Car Price Predictor") as demo:
    gr.Markdown(
        """
        # 🚗 Used Car Price Predictor
        Estimate a fair resale price from a car's brand, model, age, mileage and fuel type.
        """
    )

    with gr.Row():
        with gr.Column():
            company = gr.Dropdown(choices=COMPANIES, label="Company / Brand", value=COMPANIES[0])
            name = gr.Dropdown(
                choices=NAMES_BY_COMPANY[COMPANIES[0]],
                label="Car name / model",
                value=NAMES_BY_COMPANY[COMPANIES[0]][0],
            )
            year = gr.Slider(YEAR_MIN, YEAR_MAX, value=2015, step=1, label="Manufacturing year")
            kms_driven = gr.Slider(0, KMS_MAX, value=30000, step=1000, label="Kilometers driven")
            fuel_type = gr.Radio(choices=FUEL_TYPES, value=FUEL_TYPES[0], label="Fuel type")
            predict_btn = gr.Button("Predict price", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Estimated price", interactive=False)

    company.change(fn=update_names, inputs=company, outputs=name)
    predict_btn.click(
        fn=predict_price,
        inputs=[company, name, year, kms_driven, fuel_type],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
