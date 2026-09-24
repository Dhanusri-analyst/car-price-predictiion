{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f389cd3d-2f7a-49d2-a8c2-ace9bd70c6b7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/dhanusriprabhakaran/jupyter-env/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "* Running on local URL:  http://127.0.0.1:7860\n",
      "* To create a public link, set `share=True` in `launch()`.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"http://127.0.0.1:7860/\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pickle\n",
    "import pandas as pd\n",
    "import gradio as gr\n",
    "\n",
    "with open(\"car_price_model.pkl\", \"rb\") as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "def predict_price(company, name, year, kms_driven, fuel_type):\n",
    "    sample = pd.DataFrame([{\n",
    "        \"name\": name,\n",
    "        \"company\": company,\n",
    "        \"year\": int(year),\n",
    "        \"kms_driven\": int(kms_driven),\n",
    "        \"fuel_type\": fuel_type,\n",
    "    }])\n",
    "    predicted = model.predict(sample)[0]\n",
    "    return f\"₹ {max(predicted, 0):,.0f}\"\n",
    "\n",
    "demo = gr.Interface(\n",
    "    fn=predict_price,\n",
    "    inputs=[\n",
    "        gr.Textbox(label=\"Company (e.g. Maruti)\"),\n",
    "        gr.Textbox(label=\"Car name (e.g. Maruti Suzuki Swift)\"),\n",
    "        gr.Slider(1995, 2019, value=2015, step=1, label=\"Year\"),\n",
    "        gr.Slider(0, 400000, value=30000, step=1000, label=\"Kms driven\"),\n",
    "        gr.Radio([\"Petrol\", \"Diesel\", \"LPG\"], label=\"Fuel type\"),\n",
    "    ],\n",
    "    outputs=gr.Textbox(label=\"Predicted Price\"),\n",
    "    title=\"Used Car Price Predictor\"\n",
    ")\n",
    "\n",
    "demo.launch()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "52b1da05-8f98-4437-8918-94ccf241571b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
