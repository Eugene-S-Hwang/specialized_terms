import os
import time
import pypdf
import fitz  # PyMuPDF
import json
import subprocess
import pdfplumber
from pathlib import Path

def benchmark_pypdf(pdf_path):
    start_time = time.time()
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        end_time = time.time()
        num_pages = len(reader.pages)
        return {
            "text": text,
            "time": end_time - start_time,
            "pages": num_pages,
            "time_per_page": (end_time - start_time) / num_pages if num_pages > 0 else 0
        }
    except Exception as e:
        return {"error": str(e)}

def benchmark_pymupdf(pdf_path):
    start_time = time.time()
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            # "blocks" is better for layout preservation/multi-column
            blocks = page.get_text("blocks")
            for b in blocks:
                text += b[4] + "\n"
        end_time = time.time()
        num_pages = len(doc)
        return {
            "text": text,
            "time": end_time - start_time,
            "pages": num_pages,
            "time_per_page": (end_time - start_time) / num_pages if num_pages > 0 else 0
        }
    except Exception as e:
        return {"error": str(e)}

def benchmark_pdfplumber(pdf_path):
    start_time = time.time()
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                text += page.extract_text() or "" + "\n"
        end_time = time.time()
        return {
            "text": text,
            "time": end_time - start_time,
            "pages": num_pages,
            "time_per_page": (end_time - start_time) / num_pages if num_pages > 0 else 0
        }
    except Exception as e:
        return {"error": str(e)}

def benchmark_marker(pdf_path, output_dir):
    # Marker is a CLI tool, we'll run it and measure time
    start_time = time.time()
    try:
        # marker_single is the command for a single file
        # It creates a directory with the markdown file
        pdf_name = Path(pdf_path).stem
        marker_out = output_dir / "marker_temp"
        marker_out.mkdir(exist_ok=True)
        
        subprocess.run(
            "marker_single", 
            pdf_path, 
            "--output_dir", str(marker_out)
        , check=True, capture_output=True)
        
        end_time = time.time()
        
        # Read the resulting markdown file
        # Marker usually creates a dir named after the PDF and puts a .md file inside
        md_file = marker_out / pdf_name / f"{pdf_name}.md"
        if not md_file.exists():
            # Fallback if structure is different
            md_files = list((marker_out / pdf_name).glob("*.md"))
            if md_files:
                md_file = md_files[0]
        
        text = ""
        if md_file.exists():
            with open(md_file, "r") as f:
                text = f.read()
        
        # We need page count, use pypdf for a quick count if not available
        reader = pypdf.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        
        return {
            "text": text,
            "time": end_time - start_time,
            "pages": num_pages,
            "time_per_page": (end_time - start_time) / num_pages if num_pages > 0 else 0
        }
    except Exception as e:
        return {"error": str(e)}

def isolate_references(text):
    # Simple heuristic for references
    markers = ["References", "Bibliography", "REFERENCES", "BIBLIOGRAPHY"]
    for marker in markers:
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx:idx+500] 
    return "Not Found"

def run_benchmarks():
    pdf_dir = Path("test_extraction")
    output_dir = pdf_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    pdfs = [f for f in pdf_dir.iterdir() if f.suffix == ".pdf"]
    results = {}

    for pdf in pdfs:
        print(f"Processing {pdf.name}...")
        results[pdf.name] = {}
        
        # pypdf
        print("  Running pypdf...")
        p_res = benchmark_pypdf(str(pdf))
        if "error" not in p_res:
            results[pdf.name]["pypdf"] = {
                "time_per_page": p_res["time_per_page"],
                "ref_sample": isolate_references(p_res["text"])
            }
            with open(output_dir / f"pypdf_{pdf.stem}.txt", "w") as f:
                f.write(p_res["text"])
        
        # pymupdf
        print("  Running pymupdf...")
        m_res = benchmark_pymupdf(str(pdf))
        if "error" not in m_res:
            results[pdf.name]["pymupdf"] = {
                "time_per_page": m_res["time_per_page"],
                "ref_sample": isolate_references(m_res["text"])
            }
            with open(output_dir / f"pymupdf_{pdf.stem}.txt", "w") as f:
                f.write(m_res["text"])

        # pdfplumber
        print("  Running pdfplumber...")
        pl_res = benchmark_pdfplumber(str(pdf))
        if "error" not in pl_res:
            results[pdf.name]["pdfplumber"] = {
                "time_per_page": pl_res["time_per_page"],
                "ref_sample": isolate_references(pl_res["text"])
            }
            with open(output_dir / f"pdfplumber_{pdf.stem}.txt", "w") as f:
                f.write(pl_res["text"])

        # marker (Skipped due to high CPU overhead in benchmark)
        # print("  Running marker (this may take a while)...")
        # ma_res = benchmark_marker(str(pdf), output_dir)
        # if "error" not in ma_res:
        #     results[pdf.name]["marker"] = {
        #         "time_per_page": ma_res["time_per_page"],
        #         "ref_sample": isolate_references(ma_res["text"])
        #     }
        #     with open(output_dir / f"marker_{pdf.stem}.md", "w") as f:
        #         f.write(ma_res["text"])
        # else:
        #     print(f"    Marker failed: {ma_res['error']}")

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Benchmarks complete. Results saved to test_extraction/outputs/")

if __name__ == "__main__":
    run_benchmarks()
