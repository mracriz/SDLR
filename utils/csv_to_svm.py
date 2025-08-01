# utils/csv_to_svm.py

import pandas as pd
import argparse
from tqdm import tqdm

def convert_csv_to_svm(input_file, output_file, qid_col, relevance_col, feature_cols, no_header):
    """
    Converte um arquivo CSV para o formato SVM Rank.

    Formato de Saída:
    <relevance_label> qid:<query_id> 1:<feature_1> 2:<feature_2> ...
    """
    print(f"Lendo o arquivo CSV de entrada: {input_file}")
    
    try:
        # Se o CSV não tiver cabeçalho, os nomes das colunas precisam ser fornecidos como inteiros
        if no_header:
            # Converte nomes de colunas de string para int, se necessário
            qid_col = int(qid_col)
            relevance_col = int(relevance_col)
            feature_cols = [int(f) for f in feature_cols]
            
            # Lê o CSV sem cabeçalho
            df = pd.read_csv(input_file, header=None)
            
            # Verifica se todas as colunas de features existem
            all_cols = [qid_col, relevance_col] + feature_cols
            if not all(c in df.columns for c in all_cols):
                raise ValueError("Uma ou mais colunas especificadas não existem no CSV.")

        else:
            df = pd.read_csv(input_file)
            # Verifica se todas as colunas de features existem
            all_cols = [qid_col, relevance_col] + feature_cols
            if not all(c in df.columns for c in all_cols):
                raise ValueError("Uma ou mais colunas especificadas não existem no CSV. Verifique os nomes das colunas.")

    except FileNotFoundError:
        print(f"Erro: Arquivo de entrada não encontrado em '{input_file}'")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        return

    print("Agrupando dados por query_id...")
    # Agrupa o dataframe pelo ID da consulta para processar cada consulta individualmente
    grouped = df.groupby(qid_col)

    print(f"Iniciando a conversão para o formato SVM Rank. Total de queries: {len(grouped)}")
    
    with open(output_file, "w") as out_f:
        # Usando tqdm para mostrar uma barra de progresso
        for query_id, group in tqdm(grouped, desc="Convertendo queries"):
            for _, row in group.iterrows():
                relevance = int(row[relevance_col])
                
                # Inicia a linha com o label de relevância e o qid
                svm_line = f"{relevance} qid:{query_id}"
                
                # Adiciona as features numeradas
                feature_parts = []
                for i, col_name in enumerate(feature_cols):
                    feature_value = row[col_name]
                    # Garante que o valor da feature seja numérico e não esteja faltando
                    if pd.notna(feature_value):
                        feature_parts.append(f"{i+1}:{feature_value}")
                
                # Junta as partes da feature
                svm_line += " " + " ".join(feature_parts)
                
                # Escreve a linha completa no arquivo de saída
                out_f.write(svm_line + "\n")

    print(f"\nConversão concluída com sucesso! 🚀")
    print(f"Arquivo SVM Rank salvo em: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Converte um arquivo CSV para o formato SVM Rank.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Caminho para o arquivo CSV de entrada."
    )
    parser.add_argument(
        "--output-svm",
        required=True,
        help="Caminho para o arquivo SVM Rank de saída (ex: train.txt)."
    )
    parser.add_argument(
        "--qid-col",
        required=True,
        help="Nome da coluna que contém o Query ID (qid)."
    )
    parser.add_argument(
        "--relevance-col",
        required=True,
        help="Nome da coluna que contém o label de relevância."
    )
    parser.add_argument(
        "--feature-cols",
        required=True,
        nargs='+',
        help="Lista dos nomes das colunas de features, na ordem desejada.\nExemplo: --feature-cols feature1 feature2 feature3"
    )
    parser.add_argument(
        '--no-header',
        action='store_true',
        help='Use esta flag se o seu CSV não tiver uma linha de cabeçalho. Nesse caso, use os índices das colunas (0, 1, 2...).'
    )
    
    args = parser.parse_args()
    
    convert_csv_to_svm(
        args.input_csv,
        args.output_svm,
        args.qid_col,
        args.relevance_col,
        args.feature_cols,
        args.no_header
    )

if __name__ == "__main__":
    main()