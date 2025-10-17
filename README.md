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

The code uses python-dotenv to load `.env` automatically if present.
