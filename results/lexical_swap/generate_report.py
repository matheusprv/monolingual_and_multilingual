import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill, Border, Side
import json
import glob

wb = openpyxl.Workbook()

black_side = Side(style="thin", color="FF000000")
bottom_border = Border(bottom=black_side)

LANGUAGES = ["PB NATIVO", "ESPANHOL", "FRANCÊS", "GALEGO", "INGLÊS", "MANDARIM", "RUSSO", "ÁRABE"]
N_LANGUAGES = len(LANGUAGES) 


TASKS = ["CAPACITAR", "COMPARTILHAR", "EXPLICAR", "EXPLORAR", "FAZER", "RECOMENDAR", "RECRIAR", "RELATAR"]
N_TASKS = len(TASKS)

DECIMAL_PLACES = 2

typeLM = {
    "causalLM": {
        "files": glob.glob(f"./causalLM/*.json"),
        "metrics": ["avg_tokens_per_word", "avg_ppl", "avg_bpb"]
    },
    "maskedLM": {
        "files": glob.glob(f"./maskedLM/*.json"),
        "metrics": ["avg_tokens_per_word", "avg_pseudo_ppl", "avg_bpb"]
    },
}

def cell_color(value):
    if value < 0.0:
        return "FFF4CCCC"
    elif value > 0.0:
        return "FFB7E1CD"
    else:
        return "FFFFF2CC"


for model_type, values in typeLM.items():
    files = values["files"]
    metrics = values["metrics"]

    models = dict()
    for file_name in files:
        model = file_name.replace(".json", "")
        with open(file_name, 'r') as file:
            models[model] = json.load(file)
   
    for metric in metrics:
        ws = wb.create_sheet(title=f"{metric}_{model_type}")

        # Writting the header
        ws["A1"] = "TASK"
        ws["B1"] = "LANGUAGE"

        for k, model_name in enumerate(sorted(models.keys())):
            model_name = model_name.split("/")[-1]
            col = 3 + 2 * k
            col1 = get_column_letter(col)
            col2 = get_column_letter(col+1)

            ws[f"{col1}1"] = model_name
            ws[f"{col2}1"] = f"{model_name}_vs_PB%"


        # Writting the data
        for i, task in enumerate(TASKS):
            first_line = 2 + i * N_LANGUAGES
            last_line = first_line + N_LANGUAGES - 1  # última linha real da task

            for j, language in enumerate(LANGUAGES):
                line = first_line + j
                ws[f"A{line}"] = task
                ws[f"B{line}"] = language

                for k, (model_name, data) in enumerate(models.items()):                    
                    col = 3 + 2 * k
                    col1 = get_column_letter(col)

                    # (pseudo-)Perplexity result
                    cell = f"{col1}{line}"
                    val = data[task][language][metric]
                    ws[cell] = round(val, DECIMAL_PLACES)

                    # Comparison with PTBR
                    ptbr_val = data[task]["PB NATIVO"][metric]
                    val_comparison = (val / ptbr_val - 1) * 100

                    col2 = get_column_letter(col + 1)
                    cell = f"{col2}{line}"
                    ws[cell] = round(val_comparison, DECIMAL_PLACES)

                    color = cell_color(val_comparison)
                    ws[cell].fill = PatternFill(fgColor=color, fill_type="solid")

            # horizontal line separating tasks
            max_col = 2 + 2 * len(models)

            for col in range(1, max_col + 1):
                ws.cell(row=last_line, column=col).border = bottom_border

wb.save("report.xlsx")





