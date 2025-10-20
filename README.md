# fhir-rag-cpg
Clinical Practice Guideline with FHIR

## Setup

This project requires an OpenAI API key set in the environment as `OPENAI_API_KEY`.

On Windows PowerShell you can set it for the session with:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

Alternatively, create a `.env` file in the project root with:

```
OPENAI_API_KEY=your_key_here
```


```
Set-Location D:/HealthCare_ChatBot/fhir-rag-cpg/dataset_create
Set-Location D:\Work_LacViet\LacViet_AI\fhir-rag-cpg\dataset_create
```
Run dataset with create dataset.
```
D:/HealthCare_ChatBot/fhir-rag-cpg/venv/Scripts/python.exe .\data_genimg.py D:\HealthCare_ChatBot\fhir-rag-cpg\data\stroke_guideline.jpg
D:/Work_LacViet/LacViet_AI/fhir-rag-cpg/venv/Scripts/python.exe .\data_genimg.py  D:\Work_LacViet\LacViet_AI\fhir-rag-cpg\data\stroke_guideline.jpg

D:/Work_LacViet/LacViet_AI/fhir-rag-cpg/venv/Scripts/python.exe .\data_gentxt.py D:\Work_LacViet\LacViet_AI\fhir-rag-cpg\data\dieu-tri-chan-doan-dot-quy.txt

```