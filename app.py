from io import BytesIO
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps


TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_DIR = Path(r"C:\Program Files\Tesseract-OCR\tessdata")

if TESSERACT_EXE.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)

if TESSDATA_DIR.exists():
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


st.set_page_config(page_title="PDF 표 추출기", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.4rem !important;
        line-height: 1.15 !important;
        margin-bottom: 0.35rem !important;
    }

    h2, h3 {
        letter-spacing: 0 !important;
    }

    [data-testid="stFileUploader"] {
        margin-top: 0.35rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 116px;
        border-radius: 8px;
        border: 1px solid rgba(120, 130, 150, 0.42);
    }

    [data-testid="stFileUploaderDropzone"]::before {
        content: "파일을 이 영역에 끌어놓거나 버튼 위치를 클릭하세요";
        display: block;
        margin-bottom: 0.65rem;
        color: rgba(250, 250, 250, 0.82);
        font-weight: 650;
    }

    [data-testid="stFileUploaderDropzone"] button {
        width: 46px !important;
        min-width: 46px !important;
        height: 38px !important;
        padding: 0 !important;
        font-size: 0 !important;
        color: transparent !important;
        overflow: hidden !important;
    }

    [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stFileUploaderDropzone"] button span,
    [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] {
        font-size: 0 !important;
        color: transparent !important;
        width: 0 !important;
        overflow: hidden !important;
    }

    div[data-testid="stDownloadButton"] button,
    div[data-testid="stButton"] button {
        border-radius: 8px;
        min-height: 2.7rem;
        font-weight: 650;
    }

    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PDF/이미지 표 추출기")
st.caption("PDF나 스캔 이미지를 업로드하면 OCR 결과와 표 후보를 엑셀로 저장합니다.")

st.divider()

st.subheader("문서 업로드")
uploaded_file = st.file_uploader(
    "PDF 또는 이미지 파일",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    label_visibility="collapsed",
)

st.subheader("추출 설정")
left, middle, right = st.columns([1, 1, 1.15])

with left:
    ocr_lang = st.selectbox("OCR 언어", ["kor+eng", "kor", "eng"], index=0)

with middle:
    dpi = st.selectbox("PDF 변환 품질", [150, 200, 250, 300], index=2)

with right:
    min_columns = st.slider("표로 볼 최소 열 수", 2, 8, 3)


def load_images(file_name: str, file_bytes: bytes, pdf_dpi: int):
    extension = file_name.lower().rsplit(".", 1)[-1]

    if extension == "pdf":
        return convert_from_bytes(file_bytes, dpi=pdf_dpi)

    image = Image.open(BytesIO(file_bytes))
    return [image.convert("RGB")]


def prepare_image_for_ocr(image):
    prepared = ImageOps.grayscale(image)
    prepared = ImageOps.autocontrast(prepared)

    if prepared.width < 1800:
        ratio = 1800 / prepared.width
        prepared = prepared.resize(
            (1800, int(prepared.height * ratio)),
            Image.Resampling.LANCZOS,
        )

    prepared = prepared.filter(ImageFilter.SHARPEN)
    return prepared


def clean_ocr_words(dataframe):
    if dataframe.empty or "text" not in dataframe.columns:
        return pd.DataFrame()

    words = dataframe.copy()
    words["text"] = words["text"].fillna("").astype(str).str.strip()
    words["conf"] = pd.to_numeric(words["conf"], errors="coerce").fillna(-1)
    words = words[(words["text"] != "") & (words["text"].str.lower() != "nan")]
    words = words[words["conf"] >= 0]
    return words


def ocr_image(image, lang: str):
    config = "--psm 6 -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(image, lang=lang, config=config)


def ocr_image_data(image, lang: str):
    config = "--psm 6 -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DATAFRAME,
    )
    return clean_ocr_words(data)


def group_indices(indices, max_gap=3, min_size=2):
    if len(indices) == 0:
        return []

    groups = []
    current = [int(indices[0])]

    for index in indices[1:]:
        index = int(index)

        if index - current[-1] <= max_gap:
            current.append(index)
        else:
            if len(current) >= min_size:
                groups.append(current)
            current = [index]

    if len(current) >= min_size:
        groups.append(current)

    return [int(sum(group) / len(group)) for group in groups]


def merge_close_positions(positions, min_gap=12):
    merged = []

    for position in positions:
        if not merged or position - merged[-1] >= min_gap:
            merged.append(position)
        else:
            merged[-1] = int((merged[-1] + position) / 2)

    return merged


def max_true_run_per_row(mask):
    runs = []

    for row in mask:
        max_run = 0
        current_run = 0

        for value in row:
            if value:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        runs.append(max_run)

    return np.array(runs)


def max_true_run_per_column(mask):
    return max_true_run_per_row(mask.T)


def find_grid_lines(image):
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    array = np.array(gray)
    dark_pixels = array < 120

    row_runs = max_true_run_per_row(dark_pixels)
    column_runs = max_true_run_per_column(dark_pixels)

    row_indices = np.where(row_runs > image.width * 0.45)[0]
    column_indices = np.where(column_runs > image.height * 0.08)[0]

    rows = merge_close_positions(group_indices(row_indices), min_gap=10)
    columns = merge_close_positions(group_indices(column_indices), min_gap=10)

    return rows, columns


def clean_numeric_text(text, allow_comma=True):
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    allowed = r"[^0-9,.\-]" if allow_comma else r"[^0-9\-]"
    text = re.sub(allowed, "", text)
    text = text.strip(".,-")

    if not text:
        return ""

    if allow_comma:
        digits = re.sub(r"\D", "", text)

        if digits and len(digits) > 3 and "," not in text:
            return f"{int(digits):,}"

    return text


def clean_text_cell(text):
    text = re.sub(r"\s+", " ", text).strip(" |")
    text = re.sub(r"\b0{3,}\b", "", text)
    text = re.sub(r"(?<=[가-힣])\s*[0O]{3,}$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_note_cell(text):
    text = clean_text_cell(text)
    compact = re.sub(r"\s+", "", text)

    if not compact:
        return ""

    if re.fullmatch(r"(?i)a?4|44|aA|Aa", compact):
        return "A4"

    compact = re.sub(r"(?<=[가-힣])[A-Za-z0-9]{1,3}$", "", compact)

    if ("흑" in compact or "혹" in compact or "호" in compact) and (
        "백" in compact or "배" in compact
    ):
        return "흑백"

    if "청" in compact and ("색" in compact or "책" in compact):
        return "청색"

    if "검" in compact or "겸" in compact:
        return "검정"

    if "대" in compact and ("형" in compact or "항" in compact):
        return "대형"

    if re.search(r"[가-힣]", compact):
        compact = re.sub(r"[^가-힣A-Za-z0-9,.\-]", "", compact)
        return compact

    return re.sub(r"[^A-Za-z0-9,.\-]", "", compact)


def score_text_candidate(text, mode):
    if mode in ("integer", "money"):
        cleaned = clean_numeric_text(text, allow_comma=mode == "money")
        return len(cleaned), cleaned

    if mode == "note":
        text = clean_note_cell(text)
    else:
        text = clean_text_cell(text)

    korean_count = len(re.findall(r"[가-힣]", text))
    digit_count = len(re.findall(r"\d", text))
    alpha_count = len(re.findall(r"[A-Za-z]", text))
    punctuation_count = len(re.findall(r"[,.-]", text))
    repeated_zero_noise = len(re.findall(r"\b0{3,}\b", text))
    noise_count = len(re.findall(r"[^0-9A-Za-z가-힣,.\-\s]", text))

    if mode == "note":
        score = (
            korean_count * 3.0
            + alpha_count * 1.4
            + digit_count * 0.9
            + punctuation_count * 0.2
            - repeated_zero_noise * 5.0
            - noise_count * 2.0
        )
        return score, text

    score = (
        korean_count * 2.5
        + digit_count * 0.35
        + alpha_count * 0.7
        + punctuation_count * 0.2
        - repeated_zero_noise * 5.0
        - noise_count * 1.5
    )
    return score, text


def ocr_cell(cell_image, lang: str, mode="text"):
    cell = ImageOps.grayscale(cell_image)
    cell = ImageOps.autocontrast(cell)

    if cell.width < 520:
        ratio = 520 / cell.width
        cell = cell.resize((520, int(cell.height * ratio)), Image.Resampling.LANCZOS)

    cell = cell.filter(ImageFilter.SHARPEN)

    candidates = []
    language_candidates = [lang]
    configs = ["--psm 7", "--psm 6", "--psm 13"]

    if mode == "integer":
        language_candidates = ["eng"]
        configs = [
            "--psm 7 -c tessedit_char_whitelist=0123456789",
            "--psm 6 -c tessedit_char_whitelist=0123456789",
            "--psm 13 -c tessedit_char_whitelist=0123456789",
        ]
    elif mode == "money":
        language_candidates = ["eng"]
        configs = [
            "--psm 7 -c tessedit_char_whitelist=0123456789,.",
            "--psm 6 -c tessedit_char_whitelist=0123456789,.",
            "--psm 13 -c tessedit_char_whitelist=0123456789,.",
        ]
    elif mode == "note":
        language_candidates = []

        for candidate_lang in (lang, "kor+eng", "kor", "eng"):
            if candidate_lang not in language_candidates:
                language_candidates.append(candidate_lang)

        configs = ["--psm 7", "--psm 8", "--psm 6", "--psm 13"]

    if mode == "text":
        if "kor" in lang and "kor" not in language_candidates:
            language_candidates.append("kor")

        if "eng" in lang and "eng" not in language_candidates:
            language_candidates.append("eng")

    for ocr_lang in language_candidates:
        for config in configs:
            text = pytesseract.image_to_string(cell, lang=ocr_lang, config=config)
            text = re.sub(r"\s+", " ", text).strip(" |")

            if text:
                score, cleaned_text = score_text_candidate(text, mode)

                if cleaned_text:
                    candidates.append((score, cleaned_text))

    if not candidates:
        return ""

    return max(candidates, key=lambda item: item[0])[1]


def classify_column_mode(header_text):
    header = re.sub(r"\s+", "", str(header_text))

    if "비" in header or "고" in header:
        return "note"

    if "수" in header or "량" in header:
        return "integer"

    if "단" in header or "공급" in header or "금액" in header:
        return "money"

    return "text"


def extract_bordered_table(image, lang: str):
    rows, columns = find_grid_lines(image)

    if len(rows) < 2 or len(columns) < 2:
        return pd.DataFrame()

    cell_grid = []

    for row_index in range(len(rows) - 1):
        top = rows[row_index]
        bottom = rows[row_index + 1]

        if bottom - top < 28:
            continue

        cell_images = []

        for column_index in range(len(columns) - 1):
            left = columns[column_index]
            right = columns[column_index + 1]

            if right - left < 35:
                continue

            margin_x = max(6, int((right - left) * 0.04))
            margin_y = max(6, int((bottom - top) * 0.10))
            crop_box = (
                left + margin_x,
                top + margin_y,
                right - margin_x,
                bottom - margin_y,
            )
            cell = image.crop(crop_box)
            cell_images.append(cell)

        if cell_images:
            cell_grid.append(cell_images)

    if not cell_grid:
        return pd.DataFrame()

    header_values = [ocr_cell(cell, lang, "text") for cell in cell_grid[0]]
    column_modes = [classify_column_mode(header) for header in header_values]
    table_rows = [header_values]

    for row_cells in cell_grid[1:]:
        values = []

        for column_index, cell in enumerate(row_cells):
            mode = column_modes[column_index] if column_index < len(column_modes) else "text"
            values.append(ocr_cell(cell, lang, mode))

        if any(value.strip() for value in values):
            table_rows.append(values)

    if not table_rows:
        return pd.DataFrame()

    max_columns = max(len(row) for row in table_rows)
    padded_rows = [row + [""] * (max_columns - len(row)) for row in table_rows]
    return pd.DataFrame(padded_rows)


def parse_table_rows_from_text(text: str, min_col_count: int):
    rows = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        columns = [cell.strip() for cell in re.split(r"\s+|\t+", line) if cell.strip()]

        if len(columns) >= min_col_count:
            rows.append(columns)

    if not rows:
        return pd.DataFrame()

    max_columns = max(len(row) for row in rows)
    padded_rows = [row + [""] * (max_columns - len(row)) for row in rows]
    return pd.DataFrame(padded_rows)


def split_line_words_into_cells(line_words):
    line_words = line_words.sort_values("left")

    if len(line_words) <= 1:
        return line_words["text"].tolist()

    widths = line_words["width"].clip(lower=1)
    median_width = float(widths.median()) if not widths.empty else 20.0
    gap_threshold = max(22.0, median_width * 0.9)

    cells = []
    current_cell = []
    previous_right = None

    for _, word in line_words.iterrows():
        left = float(word["left"])
        right = left + float(word["width"])
        text = str(word["text"]).strip()

        if previous_right is not None and left - previous_right > gap_threshold:
            if current_cell:
                cells.append(" ".join(current_cell))
            current_cell = [text]
        else:
            current_cell.append(text)

        previous_right = right

    if current_cell:
        cells.append(" ".join(current_cell))

    return cells


def make_text_dataframe(page_results):
    rows = []

    for page_number, text, _ in page_results:
        for line_number, line in enumerate(text.splitlines(), start=1):
            clean_line = line.strip()

            if clean_line:
                rows.append(
                    {
                        "page": page_number,
                        "line": line_number,
                        "text": clean_line,
                    }
                )

    return pd.DataFrame(rows)


def make_coordinate_table_dataframe(page_results, min_col_count: int):
    rows = []

    for page_number, _, words in page_results:
        if words.empty:
            continue

        grouped = words.groupby(["block_num", "par_num", "line_num"], sort=True)

        for _, line_words in grouped:
            cells = split_line_words_into_cells(line_words)

            if len(cells) >= min_col_count:
                rows.append([f"p{page_number}"] + cells)

    if not rows:
        return pd.DataFrame()

    max_columns = max(len(row) for row in rows)
    padded_rows = [row + [""] * (max_columns - len(row)) for row in rows]
    return pd.DataFrame(padded_rows)


def combine_page_tables(page_tables):
    if not page_tables:
        return pd.DataFrame()

    combined_rows = []

    for page_number, dataframe in page_tables:
        if dataframe.empty:
            continue

        for _, row in dataframe.iterrows():
            combined_rows.append(row.fillna("").astype(str).tolist())

    if not combined_rows:
        return pd.DataFrame()

    max_columns = max(len(row) for row in combined_rows)
    padded_rows = [row + [""] * (max_columns - len(row)) for row in combined_rows]
    return pd.DataFrame(padded_rows)


def normalize_header_name(value, index):
    text = str(value).strip()
    compact = re.sub(r"\s+", "", text)

    if not compact:
        return f"열 {index + 1}"

    if "품" in compact:
        return "품목"

    if "수" in compact or "량" in compact:
        return "수량"

    if "단" in compact or "가" == compact:
        return "단가"

    if "공급" in compact or "금액" in compact:
        return "공급가액"

    if "비고" in compact:
        return "비고"

    return compact


def polish_table_dataframe(dataframe):
    if dataframe.empty:
        return dataframe

    table = dataframe.copy()
    table = table.fillna("").astype(str)
    table = table.loc[:, table.apply(lambda column: column.str.strip().ne("").any())]

    if table.empty:
        return table

    first_column = table.iloc[:, 0].str.strip()

    if first_column.str.fullmatch(r"p\d+").all():
        table = table.iloc[:, 1:]

    if table.empty:
        return table

    first_row = table.iloc[0].astype(str).str.strip().tolist()
    header_keywords = ("품", "수", "량", "단", "공급", "금액", "비고")
    has_header = any(any(keyword in cell for keyword in header_keywords) for cell in first_row)

    if has_header and len(table) > 1:
        headers = [normalize_header_name(value, index) for index, value in enumerate(first_row)]
        table = table.iloc[1:].reset_index(drop=True)
        table.columns = headers
    else:
        table.columns = [f"열 {index + 1}" for index in range(table.shape[1])]

    return table


def make_excel(text_df, table_df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        text_df.to_excel(writer, sheet_name="OCR_Text", index=False)

        if table_df.empty:
            pd.DataFrame({"message": ["No table-like rows found"]}).to_excel(
                writer,
                sheet_name="Parsed_Table",
                index=False,
            )
        else:
            table_df.to_excel(writer, sheet_name="Parsed_Table", index=False, header=False)

    output.seek(0)
    return output.getvalue()


if uploaded_file is not None:
    st.success(f"업로드 완료: {uploaded_file.name}")

    if st.button("표 추출하기", type="primary", use_container_width=True):
        try:
            with st.spinner("OCR로 문서를 읽는 중입니다..."):
                images = load_images(uploaded_file.name, uploaded_file.getvalue(), dpi)
                page_results = []
                page_tables = []

                for page_number, image in enumerate(images, start=1):
                    prepared_image = prepare_image_for_ocr(image)
                    text = ocr_image(prepared_image, ocr_lang)
                    words = ocr_image_data(prepared_image, ocr_lang)
                    page_results.append((page_number, text, words))

                    bordered_table = extract_bordered_table(prepared_image, ocr_lang)
                    if not bordered_table.empty:
                        page_tables.append((page_number, bordered_table))

                full_text = "\n".join(text for _, text, _ in page_results)
                text_df = make_text_dataframe(page_results)
                table_df = combine_page_tables(page_tables)

                if table_df.empty:
                    table_df = make_coordinate_table_dataframe(page_results, min_columns)

                if table_df.empty:
                    table_df = parse_table_rows_from_text(full_text, min_columns)

                table_df = polish_table_dataframe(table_df)
                excel_bytes = make_excel(text_df, table_df)

            st.success("OCR 처리가 끝났습니다.")

            result_left, result_right = st.columns([1.1, 1])

            with result_left:
                st.subheader("표 후보")
                if table_df.empty:
                    st.info("표처럼 나뉜 줄을 찾지 못했습니다. OCR_Text 시트를 확인해 주세요.")
                else:
                    st.dataframe(table_df, hide_index=True, use_container_width=True)

            with result_right:
                st.subheader("전체 OCR 텍스트")
                st.dataframe(text_df, hide_index=True, use_container_width=True)

            if table_df.empty:
                file_name = "ocr_text.xlsx"
            else:
                file_name = "extracted_tables.xlsx"

            st.download_button(
                "엑셀 다운로드",
                data=excel_bytes,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        except Exception as error:
            st.error("처리 중 오류가 발생했습니다.")
            st.exception(error)
else:
    st.info("PDF, PNG, JPG, TIF, BMP 파일을 업로드할 수 있습니다.")
