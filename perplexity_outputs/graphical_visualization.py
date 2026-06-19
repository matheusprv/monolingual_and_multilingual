import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class ResultsGraphApp:
    def __init__(
        self,
        root,
        csv_path="results_compiled.csv",
        native_language="PB NATIVO"
    ):
        self.root = root
        self.root.title("Visualizador de gráficos — Results Compiled")
        self.root.geometry("1400x850")

        self.csv_path = csv_path
        self.native_language = native_language

        self.df = None
        self.grouped = None
        self.relative = None

        self.current_fig = None
        self.canvas = None
        self.toolbar = None

        sns.set_theme(style="whitegrid")

        self.graph_options = {
            "Tokens/Word vs Perplexity": {
                "data_source": "grouped",
                "required_cols": ["Tokens/Word", "Perplexity"],
                "x": "Tokens/Word",
                "y": "Perplexity",
                "title": "Tokens/Word médio vs Perplexity média",
                "xlabel": "Tokens/Word médio",
                "ylabel": "Perplexity média",
                "filename": "tokens_word_vs_perplexity.pdf",
                "zero_lines": False,
            },
            "Tokens/Word vs Pseudo-Perplexity": {
                "data_source": "grouped",
                "required_cols": ["Tokens/Word", "Pseudo-Perplexity"],
                "x": "Tokens/Word",
                "y": "Pseudo-Perplexity",
                "title": "Tokens/Word médio vs Pseudo-Perplexity média",
                "xlabel": "Tokens/Word médio",
                "ylabel": "Pseudo-Perplexity média",
                "filename": "tokens_word_vs_pseudo_perplexity.pdf",
                "zero_lines": False,
            },
            "Tokens/Word vs BPB — modelos com Perplexity": {
                "data_source": "grouped",
                "required_cols": ["Tokens/Word", "BPB", "Perplexity"],
                "x": "Tokens/Word",
                "y": "BPB",
                "title": "Tokens/Word médio vs BPB médio — modelos com Perplexity",
                "xlabel": "Tokens/Word médio",
                "ylabel": "BPB médio",
                "filename": "tokens_word_vs_bpb_perplexity_models.pdf",
                "zero_lines": False,
            },
            "Tokens/Word vs BPB — modelos com Pseudo-Perplexity": {
                "data_source": "grouped",
                "required_cols": ["Tokens/Word", "BPB", "Pseudo-Perplexity"],
                "x": "Tokens/Word",
                "y": "BPB",
                "title": "Tokens/Word médio vs BPB médio — modelos com Pseudo-Perplexity",
                "xlabel": "Tokens/Word médio",
                "ylabel": "BPB médio",
                "filename": "tokens_word_vs_bpb_pseudo_perplexity_models.pdf",
                "zero_lines": False,
            },
            "Variação % vs PB NATIVO — Perplexity": {
                "data_source": "relative",
                "required_cols": ["Tokens/Word", "Perplexity_%"],
                "x": "Tokens/Word",
                "y": "Perplexity_%",
                "title": "Variação percentual vs PB NATIVO — Perplexity",
                "xlabel": "Tokens/Word",
                "ylabel": "Variação percentual da Perplexity (%)",
                "filename": "relative_tokens_word_vs_perplexity.pdf",
                "zero_lines": True,
            },
            "Variação % vs PB NATIVO — Pseudo-Perplexity": {
                "data_source": "relative",
                "required_cols": ["Tokens/Word", "Pseudo-Perplexity_%"],
                "x": "Tokens/Word",
                "y": "Pseudo-Perplexity_%",
                "title": "Variação percentual vs PB NATIVO — Pseudo-Perplexity",
                "xlabel": "Tokens/Word",
                "ylabel": "Variação percentual da Pseudo-Perplexity (%)",
                "filename": "relative_tokens_word_vs_pseudo_perplexity.pdf",
                "zero_lines": True,
            },
            "Variação % vs PB NATIVO — BPB em modelos com Perplexity": {
                "data_source": "relative",
                "required_cols": ["Tokens/Word", "BPB_%", "Perplexity"],
                "x": "Tokens/Word",
                "y": "BPB_%",
                "title": "Variação percentual vs PB NATIVO — BPB em modelos com Perplexity",
                "xlabel": "Tokens/Word",
                "ylabel": "Variação percentual do BPB (%)",
                "filename": "relative_tokens_word_vs_bpb_perplexity_models.pdf",
                "zero_lines": True,
            },
            "Variação % vs PB NATIVO — BPB em modelos com Pseudo-Perplexity": {
                "data_source": "relative",
                "required_cols": ["Tokens/Word", "BPB_%", "Pseudo-Perplexity"],
                "x": "Tokens/Word",
                "y": "BPB_%",
                "title": "Variação percentual vs PB NATIVO — BPB em modelos com Pseudo-Perplexity",
                "xlabel": "Tokens/Word",
                "ylabel": "Variação percentual do BPB (%)",
                "filename": "relative_tokens_word_vs_bpb_pseudo_perplexity_models.pdf",
                "zero_lines": True,
            },
        }

        self.load_data()
        self.build_interface()
        self.populate_filters()
        self.plot_selected_graph()

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)

        numeric_cols = [
            "Tokens/Word",
            "Perplexity",
            "Pseudo-Perplexity",
            "BPB"
        ]

        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.grouped = (
            self.df
            .groupby(["Model", "Language"], as_index=False)
            .agg({
                "Tokens/Word": "mean",
                "Perplexity": "mean",
                "Pseudo-Perplexity": "mean",
                "BPB": "mean"
            })
        )

        native = self.grouped[
            self.grouped["Language"] == self.native_language
        ].copy()

        if native.empty:
            raise ValueError(
                f"Nenhuma linha encontrada com Language == '{self.native_language}'. "
                "Verifique o nome exato da língua nativa no CSV."
            )

        native = native.rename(columns={
            "Tokens/Word": "Tokens/Word_Native",
            "Perplexity": "Perplexity_Native",
            "Pseudo-Perplexity": "Pseudo-Perplexity_Native",
            "BPB": "BPB_Native",
        })

        native = native[
            [
                "Model",
                "Tokens/Word_Native",
                "Perplexity_Native",
                "Pseudo-Perplexity_Native",
                "BPB_Native",
            ]
        ]

        self.relative = self.grouped.merge(native, on="Model", how="left")

        self.relative["Tokens/Word_%"] = (
            self.relative["Tokens/Word"] /
            self.relative["Tokens/Word_Native"]
            - 1
        ) * 100

        self.relative["Perplexity_%"] = (
            self.relative["Perplexity"] /
            self.relative["Perplexity_Native"]
            - 1
        ) * 100

        self.relative["Pseudo-Perplexity_%"] = (
            self.relative["Pseudo-Perplexity"] /
            self.relative["Pseudo-Perplexity_Native"]
            - 1
        ) * 100

        self.relative["BPB_%"] = (
            self.relative["BPB"] /
            self.relative["BPB_Native"]
            - 1
        ) * 100

    def build_interface(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_frame, padding=10)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y)

        graph_frame = ttk.Frame(main_frame, padding=10)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.graph_frame = graph_frame

        ttk.Label(
            controls_frame,
            text="Escolha o gráfico:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w", pady=(0, 5))

        self.graph_var = tk.StringVar()
        self.graph_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.graph_var,
            values=list(self.graph_options.keys()),
            state="readonly",
            width=45
        )
        self.graph_combo.current(0)
        self.graph_combo.pack(anchor="w", pady=(0, 15))
        self.graph_combo.bind(
            "<<ComboboxSelected>>",
            lambda event: self.on_graph_change()
        )

        ttk.Label(
            controls_frame,
            text="Línguas incluídas:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w")

        self.language_listbox = tk.Listbox(
            controls_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=10,
            width=45
        )
        self.language_listbox.pack(anchor="w", pady=(5, 5))

        lang_buttons = ttk.Frame(controls_frame)
        lang_buttons.pack(anchor="w", pady=(0, 15))

        ttk.Button(
            lang_buttons,
            text="Selecionar todas",
            command=self.select_all_languages_and_update_models
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            lang_buttons,
            text="Limpar",
            command=self.clear_languages_and_update_models
        ).pack(side=tk.LEFT)

        ttk.Label(
            controls_frame,
            text="Modelos incluídos:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w")

        self.model_listbox = tk.Listbox(
            controls_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=14,
            width=45
        )
        self.model_listbox.pack(anchor="w", pady=(5, 5))

        model_buttons = ttk.Frame(controls_frame)
        model_buttons.pack(anchor="w", pady=(0, 15))

        ttk.Button(
            model_buttons,
            text="Selecionar todos",
            command=self.select_all_models
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            model_buttons,
            text="Limpar",
            command=self.clear_models
        ).pack(side=tk.LEFT)

        ttk.Button(
            controls_frame,
            text="Atualizar gráfico",
            command=self.plot_selected_graph
        ).pack(fill=tk.X, pady=(5, 5))

        ttk.Button(
            controls_frame,
            text="Salvar gráfico atual como PDF",
            command=self.save_current_graph_pdf
        ).pack(fill=tk.X, pady=(5, 5))

        ttk.Button(
            controls_frame,
            text="Carregar outro CSV",
            command=self.load_another_csv
        ).pack(fill=tk.X, pady=(5, 5))

        self.status_var = tk.StringVar()
        self.status_var.set("Pronto.")

        ttk.Label(
            controls_frame,
            textvariable=self.status_var,
            wraplength=320
        ).pack(anchor="w", pady=(20, 0))

    def select_all_languages_and_update_models(self):
        self.select_all_languages()
        self.update_model_filter_for_selected_graph()

    def clear_languages_and_update_models(self):
        self.clear_languages()
        self.update_model_filter_for_selected_graph()

    def populate_filters(self):
        self.language_listbox.delete(0, tk.END)
        self.model_listbox.delete(0, tk.END)

        languages = sorted(self.grouped["Language"].dropna().unique())

        for lang in languages:
            self.language_listbox.insert(tk.END, lang)

        self.select_all_languages()
        self.update_model_filter_for_selected_graph()

    def on_graph_change(self):
        self.update_model_filter_for_selected_graph()
        self.plot_selected_graph()

    def update_model_filter_for_selected_graph(self):
        graph_name = self.graph_var.get()
        graph_config = self.graph_options[graph_name]

        if graph_config["data_source"] == "grouped":
            data = self.grouped.copy()
        else:
            data = self.relative.copy()

        # Mantém apenas as linhas que possuem os valores necessários
        data = data.dropna(subset=graph_config["required_cols"])

        # Opcional: respeita as línguas já selecionadas
        selected_languages = self.get_selected_languages()

        if selected_languages:
            data = data[data["Language"].isin(selected_languages)]

        valid_models = sorted(data["Model"].dropna().unique())

        # Guarda os modelos selecionados antes de atualizar a lista
        previously_selected_models = set(self.get_selected_models())

        self.model_listbox.delete(0, tk.END)

        for model in valid_models:
            self.model_listbox.insert(tk.END, model)

        # Tenta preservar os modelos que já estavam selecionados
        # desde que ainda sejam válidos para o novo gráfico
        for i, model in enumerate(valid_models):
            if model in previously_selected_models:
                self.model_listbox.select_set(i)

        # Se nenhum modelo ficou selecionado, seleciona todos os válidos
        if not self.model_listbox.curselection():
            self.select_all_models()

    def select_all_languages(self):
        self.language_listbox.select_set(0, tk.END)

    def clear_languages(self):
        self.language_listbox.select_clear(0, tk.END)

    def select_all_models(self):
        self.model_listbox.select_set(0, tk.END)

    def clear_models(self):
        self.model_listbox.select_clear(0, tk.END)

    def get_selected_languages(self):
        indices = self.language_listbox.curselection()
        return [self.language_listbox.get(i) for i in indices]

    def get_selected_models(self):
        indices = self.model_listbox.curselection()
        return [self.model_listbox.get(i) for i in indices]

    def get_filtered_data(self, graph_config):
        if graph_config["data_source"] == "grouped":
            data = self.grouped.copy()
        else:
            data = self.relative.copy()

        selected_languages = self.get_selected_languages()
        selected_models = self.get_selected_models()

        if not selected_languages:
            raise ValueError("Selecione pelo menos uma língua.")

        if not selected_models:
            raise ValueError("Selecione pelo menos um modelo.")

        data = data[
            data["Language"].isin(selected_languages) &
            data["Model"].isin(selected_models)
        ]

        data = data.dropna(subset=graph_config["required_cols"])

        return data

    def plot_selected_graph(self):
        graph_name = self.graph_var.get()
        graph_config = self.graph_options[graph_name]

        try:
            data = self.get_filtered_data(graph_config)
        except ValueError as error:
            messagebox.showwarning("Filtro inválido", str(error))
            return

        if data.empty:
            messagebox.showwarning(
                "Sem dados",
                "Não há dados disponíveis para esse gráfico com os filtros atuais."
            )
            return

        print(f"\nDados plotados em '{graph_name}':")
        print(data.to_string(index=False))
        print(f"Total de pontos: {len(data)}\n", flush=True)

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        if self.toolbar is not None:
            self.toolbar.destroy()

        if self.current_fig is not None:
            plt.close(self.current_fig)

        self.current_fig, ax = plt.subplots(figsize=(10, 6))

        sns.scatterplot(
            data=data,
            x=graph_config["x"],
            y=graph_config["y"],
            hue="Language",
            style="Model",
            s=80,
            ax=ax
        )

        if graph_config["zero_lines"]:
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.axvline(0, linestyle="--", linewidth=1)

        ax.set_title(graph_config["title"])
        ax.set_xlabel(graph_config["xlabel"])
        ax.set_ylabel(graph_config["ylabel"])

        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0
        )

        self.current_fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.current_fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()

        self.status_var.set(
            f"Exibindo: {graph_name}\n"
            f"Pontos no gráfico: {len(data)}"
        )

    def save_current_graph_pdf(self):
        if self.current_fig is None:
            messagebox.showwarning(
                "Nenhum gráfico",
                "Nenhum gráfico foi gerado ainda."
            )
            return

        graph_name = self.graph_var.get()
        default_filename = self.graph_options[graph_name]["filename"]

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=default_filename,
            filetypes=[
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.current_fig.savefig(
            file_path,
            format="pdf",
            bbox_inches="tight"
        )

        messagebox.showinfo(
            "Gráfico salvo",
            f"Gráfico salvo com sucesso em:\n{file_path}"
        )

    def load_another_csv(self):
        file_path = filedialog.askopenfilename(
            title="Selecione o arquivo CSV",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            self.csv_path = file_path
            self.load_data()
            self.populate_filters()
            self.plot_selected_graph()
            self.status_var.set(f"Arquivo carregado: {file_path}")
        except Exception as error:
            messagebox.showerror(
                "Erro ao carregar CSV",
                str(error)
            )


def main():
    root = tk.Tk()

    app = ResultsGraphApp(
        root=root,
        csv_path="results_compiled.csv",
        native_language="PB NATIVO"
    )

    root.mainloop()


if __name__ == "__main__":
    main()