"""
Enhanced Clinical Practice Guideline (CPG) Builder with Vietnamese Support
Converts medical documents (PDF, DOC, DOCX, TXT) to comprehensive FHIR CPG resources
Supports Vietnamese translation and optimized Arden Syntax to CQL conversion
"""

from fhir.resources.plandefinition import PlanDefinition
from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.questionnaire import Questionnaire, QuestionnaireItem
from fhir.resources.measure import Measure, MeasureGroup, MeasureGroupPopulation
from fhir.resources.evidence import Evidence
from fhir.resources.evidencevariable import EvidenceVariable
from fhir.resources.measurereport import MeasureReport
from fhir.resources.bundle import Bundle, BundleEntry
from fhir.resources.library import Library
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.relatedartifact import RelatedArtifact
from datetime import datetime
import re
import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io

load_dotenv()

class LanguageDetector:
    """Detect and translate Vietnamese text"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Vietnamese or English"""
        # Simple heuristic: check for Vietnamese characters
        vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
        text_lower = text.lower()
        
        vietnamese_count = sum(1 for char in text_lower if char in vietnamese_chars)
        total_alpha = sum(1 for char in text_lower if char.isalpha())
        
        if total_alpha == 0:
            return "unknown"
        
        vietnamese_ratio = vietnamese_count / total_alpha
        
        if vietnamese_ratio > 0.1:  # If >10% Vietnamese chars
            return "vi"
        return "en"
    
    def translate_to_vietnamese(self, text: str, text_type: str = "general") -> str:
        """Translate English text to Vietnamese using GPT"""
        print(f"   🌐 Translating {text_type} to Vietnamese...")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical translator specializing in clinical terminology. Translate accurately while preserving medical terms and maintaining professional clinical language."
                    },
                    {
                        "role": "user",
                        "content": f"Translate this medical text to Vietnamese. Preserve medical codes, measurements, and clinical terminology:\n\n{text}"
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            translated = response.choices[0].message.content.strip()
            print(f"   ✓ Translation complete")
            return translated
            
        except Exception as e:
            print(f"   ⚠️  Translation failed: {e}")
            return text


class MedicalDocumentExtractor:
    """Extract structured medical data from documents using LLM and OCR"""
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None):
        self.llm_provider = llm_provider
        self.client = None
        self.translator = None
        
        if llm_provider == "openai":
            try:
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found")
                self.client = OpenAI(api_key=api_key)
                self.translator = LanguageDetector(self.client)
            except ImportError:
                raise ImportError("OpenAI not available. Install: pip install openai")
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF, using OCR if necessary"""
        print(f"📄 Processing PDF: {Path(pdf_path).name}")
        
        text = ""
        
        # Try to extract text directly
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + "\n"
                    else:
                        print(f"   ⚠️  Page {page_num + 1} has no extractable text, using OCR...")
        except Exception as e:
            print(f"   ⚠️  Text extraction failed: {e}")
        
        # If no text extracted, use OCR
        if not text.strip():
            print("   🔍 Using OCR to extract text from scanned PDF...")
            try:
                images = convert_from_path(pdf_path, dpi=300)
                for i, image in enumerate(images):
                    print(f"      Processing page {i + 1}/{len(images)}...")
                    # Try Vietnamese OCR first
                    page_text = pytesseract.image_to_string(image, lang='vie+eng')
                    text += page_text + "\n"
            except Exception as e:
                raise Exception(f"OCR extraction failed: {e}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from PDF")
        
        print(f"   ✓ Extracted {len(text)} characters")
        return text.strip()
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        print(f"📄 Processing DOCX: {Path(docx_path).name}")
        
        try:
            import docx
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            if not text.strip():
                raise ValueError("No text found in DOCX file")
            
            print(f"   ✓ Extracted {len(text)} characters")
            return text.strip()
        except ImportError:
            raise ImportError("python-docx not installed. Install: pip install python-docx")
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {e}")
    
    def extract_text_from_doc(self, doc_path: str) -> str:
        """Extract text from legacy DOC file"""
        print(f"📄 Processing DOC: {Path(doc_path).name}")
        
        try:
            import textract
            text = textract.process(doc_path).decode('utf-8')
            
            if not text.strip():
                raise ValueError("No text found in DOC file")
            
            print(f"   ✓ Extracted {len(text)} characters")
            return text.strip()
        except ImportError:
            raise ImportError("textract not installed. Install: pip install textract")
        except Exception as e:
            raise Exception(f"DOC extraction failed: {e}")
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        print(f"📄 Processing TXT: {Path(txt_path).name}")
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            if not text.strip():
                raise ValueError("Text file is empty")
            
            print(f"   ✓ Extracted {len(text)} characters")
            return text.strip()
        except UnicodeDecodeError:
            # Try different encoding
            with open(txt_path, 'r', encoding='latin-1') as file:
                text = file.read()
            print(f"   ✓ Extracted {len(text)} characters (latin-1 encoding)")
            return text.strip()
        except Exception as e:
            raise Exception(f"TXT extraction failed: {e}")
    
    def extract_text_from_document(self, document_path: str) -> str:
        """Extract text from any supported document format"""
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        ext = Path(document_path).suffix.lower()
        
        extractors = {
            '.pdf': self.extract_text_from_pdf,
            '.docx': self.extract_text_from_docx,
            '.doc': self.extract_text_from_doc,
            '.txt': self.extract_text_from_txt
        }
        
        if ext not in extractors:
            raise ValueError(f"Unsupported file format: {ext}. Supported: {', '.join(extractors.keys())}")
        
        return extractors[ext](document_path)
    
    def create_extraction_prompt(self) -> str:
        """Create detailed prompt for medical document extraction"""
        return """Analyze this medical clinical practice guideline document and extract ALL clinical decision logic.

Return a JSON object with this structure:

{
  "metadata": {
    "title": "Full guideline title",
    "disease": "Primary disease/condition name",
    "domain": "Medical domain",
    "purpose": "Clinical purpose",
    "target_population": "Target patient population",
    "evidence_level": "A|B|C",
    "guideline_source": "Source organization"
  },
  "entry_point": {
    "id": "entry",
    "condition": "Initial presenting condition",
    "inclusion_criteria": ["criteria1", "criteria2"],
    "exclusion_criteria": ["criteria1", "criteria2"]
  },
  "decision_nodes": [
    {
      "id": "unique_id",
      "type": "condition|action|assessment",
      "label": "Node label",
      "question": "Decision question",
      "condition_expression": "Logical condition",
      "measurement": {
        "parameter": "What is measured",
        "unit": "Unit of measurement",
        "normal_range": "Normal values",
        "threshold": "Decision threshold"
      },
      "branches": {
        "yes": {
          "next_node": "next_id",
          "actions": ["action1"],
          "description": "Description"
        },
        "no": {
          "next_node": "next_id",
          "actions": ["action1"],
          "description": "Description"
        }
      },
      "timing": "Timing constraint",
      "priority": "urgent|routine|stat",
      "clinical_codes": {
        "snomed": ["code1"],
        "icd10": ["code1"],
        "loinc": ["code1"]
      },
      "evidence": {
        "quality": "high|moderate|low",
        "recommendation_strength": "strong|weak",
        "references": ["ref1"]
      }
    }
  ],
  "action_nodes": [
    {
      "id": "action_id",
      "type": "medication|procedure|lab_test|imaging|consult",
      "title": "Action title",
      "description": "Description",
      "instructions": "Step-by-step instructions",
      "dosage": "Medication dosage if applicable",
      "route": "Administration route",
      "frequency": "How often",
      "duration": "How long",
      "contraindications": ["contra1"],
      "precautions": ["precaution1"],
      "monitoring": ["what to monitor"],
      "clinical_codes": {
        "rxnorm": ["code"],
        "cpt": ["code"],
        "snomed": ["code"]
      }
    }
  ],
  "assessment_questions": [
    {
      "id": "question_id",
      "text": "Question text",
      "type": "boolean|choice|decimal|integer|string",
      "required": true,
      "options": ["option1", "option2"],
      "linked_node": "decision_node_id"
    }
  ],
  "quality_measures": [
    {
      "id": "measure_id",
      "title": "Measure title",
      "description": "What is measured",
      "type": "process|outcome|structure",
      "numerator": "Numerator definition",
      "denominator": "Denominator definition",
      "improvement_notation": "increase|decrease"
    }
  ],
  "endpoints": [
    {
      "id": "endpoint_id",
      "category": "diagnosis|treatment|referral",
      "diagnosis": "Final diagnosis",
      "recommendations": ["rec1"],
      "follow_up": "Follow-up instructions",
      "expected_outcome": "Expected result"
    }
  ]
}

Extract EVERY clinical decision point, condition, measurement value, and clinical action from the document.
Return ONLY valid JSON."""
    
    def extract_from_document(self, document_path: str, translate_to_vietnamese: bool = True) -> Dict:
        """Extract structured clinical data from medical document"""
        print(f"📊 Analyzing medical document: {Path(document_path).name}")
        
        # Extract text from document
        document_text = self.extract_text_from_document(document_path)
        
        # Detect language
        detected_lang = self.translator.detect_language(document_text[:1000])
        print(f"   🌐 Detected language: {detected_lang}")
        
        # Translate if needed
        original_text = document_text
        if detected_lang == "en" and translate_to_vietnamese:
            print("   🔄 Document is in English, translating to Vietnamese...")
            document_text = self.translator.translate_to_vietnamese(document_text, "medical document")
        elif detected_lang == "vi":
            print("   ✓ Document is already in Vietnamese, no translation needed")
        
        # Save extracted text for reference
        output_dir = Path("cpg_output") / "temp"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "extracted_text_original.txt", 'w', encoding='utf-8') as f:
            f.write(original_text)
        
        if original_text != document_text:
            with open(output_dir / "extracted_text_vietnamese.txt", 'w', encoding='utf-8') as f:
                f.write(document_text)
            print(f"   💾 Saved: extracted_text_original.txt and extracted_text_vietnamese.txt")
        else:
            with open(output_dir / "extracted_text.txt", 'w', encoding='utf-8') as f:
                f.write(document_text)
            print(f"   💾 Saved: extracted_text.txt")
        
        extraction_prompt = self.create_extraction_prompt()
        
        try:
            # Split text if too long
            max_chars = 100000
            if len(document_text) > max_chars:
                print(f"   ⚠️  Document is large ({len(document_text)} chars), processing in chunks...")
                document_text = document_text[:max_chars]
                print(f"   📝 Processing first {max_chars} characters")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical informatics expert specializing in extracting structured clinical decision logic from medical guidelines."
                    },
                    {
                        "role": "user",
                        "content": f"{extraction_prompt}\n\n=== DOCUMENT TEXT ===\n\n{document_text}"
                    }
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            extracted_data = json.loads(content)
            
            # Add language metadata
            extracted_data['language'] = {
                'detected': detected_lang,
                'translated': detected_lang == "en" and translate_to_vietnamese
            }
            
            print(f"✅ Extraction complete!")
            print(f"   - Decision nodes: {len(extracted_data.get('decision_nodes', []))}")
            print(f"   - Action nodes: {len(extracted_data.get('action_nodes', []))}")
            print(f"   - Questions: {len(extracted_data.get('assessment_questions', []))}")
            print(f"   - Measures: {len(extracted_data.get('quality_measures', []))}")
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw response: {content[:500]}...")
            raise
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            raise


class ArdenSyntaxGenerator:
    """Generate Arden Syntax MLM from extracted data"""
    
    def generate(self, extracted_data: Dict, plan_def: PlanDefinition) -> str:
        """Generate Arden Syntax Medical Logic Module"""
        
        print("⚕️  Generating Arden Syntax MLM...")
        
        # Safe access to title
        title = getattr(plan_def, 'title', 'Clinical_Guideline')
        if not title:
            title = "Clinical_Guideline"
        title = title.replace(" ", "_")
        
        plan_id = getattr(plan_def, 'id', 'guideline')
        version = getattr(plan_def, 'version', '1.0.0')
        description = getattr(plan_def, 'description', 'Clinical decision support guideline')
        
        metadata = extracted_data.get("metadata", {})
        decision_nodes = extracted_data.get("decision_nodes", [])
        entry = extracted_data.get("entry_point", {})
        
        # Get topics safely
        topics = []
        if hasattr(plan_def, 'topic') and plan_def.topic:
            for topic in plan_def.topic:
                if hasattr(topic, 'text') and topic.text:
                    topics.append(topic.text)
        
        keywords = ', '.join(topics) if topics else 'clinical decision support'
        
        mlm = f"""maintenance:
    title: {title};;
    mlmname: {plan_id};;
    arden: Version 2.10;;
    version: {version};;
    institution: Generated from Medical Document;;
    author: Automated Conversion System;;
    specialist: ;;
    date: {datetime.now().strftime('%Y-%m-%d')};;
    validation: testing;;

library:
    purpose: {description};;
    explanation: This MLM implements clinical decision logic extracted from
                 a medical guideline document using AI text analysis and
                 converted to FHIR PlanDefinition format.;;
    keywords: {keywords};;
    citations: ;;

knowledge:
    type: data-driven;;
    
    data:
        /* Patient identifiers */
        patient_id := read {{patient_identifier}};
        encounter_id := read {{encounter_id}};
        
        /* Clinical data elements from document */
"""
        
        # Add data elements from decision nodes
        for node in decision_nodes:
            if node.get('condition_expression'):
                var_name = re.sub(r'[^a-zA-Z0-9_]', '_', node['id'])
                mlm += f"        {var_name} := read {{{node['condition_expression']}}};\n"
        
        mlm += """    ;;
    
    evoke:
        /* Trigger conditions */
"""
        
        # Add evoke from entry point
        if entry.get('condition'):
            mlm += f"        {entry['condition']} OR\n"
        mlm += "        clinical_decision_support_requested\n"
        
        mlm += """    ;;
    
    logic:
        /* Initialize */
        recommendations := [];
        decision_path := "";
        
        /* Implement decision tree logic */
"""
        
        # Generate logic for each decision node
        for idx, node in enumerate(decision_nodes):
            label = node.get('label', f'Decision {idx}')
            mlm += f"\n        /* {label} */\n"
            
            var_name = re.sub(r'[^a-zA-Z0-9_]', '_', node['id'])
            
            if node.get('condition_expression'):
                mlm += f"        if {var_name} then\n"
                
                if node.get('branches', {}).get('yes', {}).get('actions'):
                    for action in node['branches']['yes']['actions']:
                        mlm += f'            recommendations := recommendations + ["{action}"];\n'
                        mlm += f'            decision_path := decision_path + " -> {node["id"]} (YES)";\n'
                
                mlm += "        else\n"
                
                if node.get('branches', {}).get('no', {}).get('actions'):
                    for action in node['branches']['no']['actions']:
                        mlm += f'            recommendations := recommendations + ["{action}"];\n'
                        mlm += f'            decision_path := decision_path + " -> {node["id"]} (NO)";\n'
                
                mlm += "        endif;\n"
        
        mlm += """
        /* Prepare output */
        if count(recommendations) > 0 then
            alert_needed := true;
            alert_message := "Clinical Decision Support: " || 
                          count(recommendations) || " recommendations";
        else
            alert_needed := false;
            alert_message := "No specific recommendations at this time";
        endif;
    ;;
    
    action:
        /* Execute recommendations */
        if alert_needed then
            write alert_message;
            write "Decision Path: " || decision_path;
            
            for rec in recommendations do
                write "  - " || rec;
            enddo;
            
            return recommendations;
        endif;
    ;;

end:
"""
        
        print("   ✓ Arden Syntax MLM generated")
        return mlm


class CQLGenerator:
    """Generate CQL from Arden Syntax MLM"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def generate_from_arden(self, arden_mlm: str, extracted_data: Dict, plan_def: PlanDefinition) -> str:
        """Generate optimized CQL from Arden Syntax using GPT"""
        
        print("🔧 Generating CQL from Arden Syntax...")
        
        try:
            prompt = f"""Convert this Arden Syntax MLM to Clinical Quality Language (CQL).

Requirements:
1. Use FHIR version 4.0.1
2. Include FHIRHelpers library
3. Create proper value sets and codes
4. Implement all decision logic from the Arden MLM
5. Add proper context (Patient)
6. Create reusable definitions
7. Follow CQL best practices

Arden Syntax MLM:
{arden_mlm}

Generate complete, valid CQL code."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in clinical decision support languages, specializing in converting Arden Syntax to CQL."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            cql_content = response.choices[0].message.content
            
            # Extract CQL from markdown if present
            cql_match = re.search(r'```(?:cql)?\s*(.*?)\s*```', cql_content, re.DOTALL)
            if cql_match:
                cql_content = cql_match.group(1)
            
            # Ensure proper library declaration
            plan_name = getattr(plan_def, 'name', 'ClinicalGuideline')
            if not cql_content.strip().startswith('library'):
                cql_content = f"library {plan_name} version '1.0.0'\n\n" + cql_content
            
            print("   ✓ CQL generated from Arden Syntax")
            return cql_content
            
        except Exception as e:
            print(f"   ⚠️  CQL generation from Arden failed: {e}")
            # Fallback to basic CQL
            return self._generate_basic_cql(extracted_data, plan_def)
    
    def _generate_basic_cql(self, extracted_data: Dict, plan_def: PlanDefinition) -> str:
        """Generate basic CQL as fallback"""
        metadata = extracted_data.get("metadata", {})
        decision_nodes = extracted_data.get("decision_nodes", [])
        entry = extracted_data.get("entry_point", {})
        
        plan_name = getattr(plan_def, 'name', 'ClinicalGuideline')
        title = getattr(plan_def, 'title', 'Clinical Guideline')
        
        cql_lines = [
            f"library {plan_name} version '1.0.0'",
            "",
            "using FHIR version '4.0.1'",
            "",
            "include FHIRHelpers version '4.0.1'",
            "",
            f"// {title}",
            f"// Domain: {metadata.get('domain', 'N/A')}",
            f"// Disease: {metadata.get('disease', 'N/A')}",
            "",
            "context Patient",
            "",
            "// =============================================================",
            "// Data Elements",
            "// =============================================================",
            'define "Patient Age":',
            "  AgeInYears()",
            "",
            'define "Patient Gender":',
            "  Patient.gender",
            "",
            "// =============================================================",
            "// Inclusion Criteria",
            "// =============================================================",
        ]
        
        if entry.get("inclusion_criteria"):
            cql_lines.append('define "Meets Inclusion Criteria":')
            for criteria in entry["inclusion_criteria"]:
                cql_lines.append(f"  // {criteria}")
            cql_lines.append("  true // TODO: Implement actual logic")
        else:
            cql_lines.append('define "Meets Inclusion Criteria":\n  true')
        
        cql_lines += [
            "",
            "// =============================================================",
            "// Decision Logic",
            "// =============================================================",
        ]
        
        for node in decision_nodes:
            var_name = node.get("id", "Node").replace("-", "_")
            var_name = "".join(word.capitalize() for word in var_name.split("_"))
            question = node.get("question", "")
            condition = node.get("condition_expression", "")
            
            cql_lines.append(f'\ndefine "{var_name}":')
            if question:
                cql_lines.append(f"  // {question}")
            if condition:
                cql_lines.append(f"  // Condition: {condition}")
            cql_lines.append("  false // TODO: Implement logic")
        
        cql_lines += [
            "",
            "// =============================================================",
            "// Recommendations",
            "// =============================================================",
            'define "Active Recommendations":',
            "  {} // TODO: Implement recommendations",
        ]
        
        return "\n".join(cql_lines) + "\n"


class FHIRCPGConverter:
    """Convert extracted medical data to comprehensive FHIR CPG resources"""
    
    def __init__(self):
        self.base_url = "http://example.org/fhir"
    
    def sanitize_id(self, text: str) -> str:
        """Convert text to valid FHIR ID"""
        if not text:
            return "default_id"
        return re.sub(r'[^a-zA-Z0-9-]', '-', text.lower())[:64]
    
    def create_plan_definition(self, extracted_data: Dict) -> PlanDefinition:
        """Create comprehensive FHIR PlanDefinition"""
        metadata = extracted_data.get('metadata', {})
        title = metadata.get('title', 'Clinical Decision Algorithm')
        plan_id = self.sanitize_id(title)
        
        print(f"📝 Creating PlanDefinition: {title}")
        
        actions = self._build_actions_from_nodes(
            extracted_data.get('decision_nodes', []),
            extracted_data.get('action_nodes', [])
        )
        
        plan_def = PlanDefinition.construct(
            id=plan_id,
            url=f"{self.base_url}/PlanDefinition/{plan_id}",
            identifier=[{
                "system": f"{self.base_url}/identifiers",
                "value": plan_id
            }],
            version="1.0.0",
            name=self.sanitize_id(title).replace('-', '_'),
            title=title,
            type=CodeableConcept.construct(
                coding=[Coding.construct(
                    system="http://terminology.hl7.org/CodeSystem/plan-definition-type",
                    code="clinical-protocol",
                    display="Clinical Protocol"
                )]
            ),
            status="draft",
            experimental=True,
            date=datetime.now().isoformat(),
            publisher=metadata.get('guideline_source', 'Medical Document Converter'),
            description=metadata.get('purpose', 'Clinical decision support protocol'),
            purpose=metadata.get('purpose'),
            usage=f"Target Population: {metadata.get('target_population', 'Not specified')}",
            useContext=[{
                "code": Coding.construct(
                    system="http://terminology.hl7.org/CodeSystem/usage-context-type",
                    code="focus",
                    display="Clinical Focus"
                ),
                "valueCodeableConcept": CodeableConcept.construct(
                    text=metadata.get('domain', 'Clinical Medicine')
                )
            }],
            topic=[
                CodeableConcept.construct(
                    text=metadata.get('disease', metadata.get('domain', 'Clinical Decision Support'))
                )
            ],
            action=actions
        )
        
        print(f"   ✓ Created {len(actions)} top-level actions")
        return plan_def
    
    def _build_actions_from_nodes(self, decision_nodes: List[Dict], 
                                   action_nodes: List[Dict]) -> List[Dict]:
        """Build FHIR action hierarchy from decision and action nodes"""
        actions = []
        action_node_map = {node['id']: node for node in action_nodes}
        
        for node in decision_nodes:
            action = self._create_action_from_node(node, action_node_map)
            if action:
                actions.append(action)
        
        return actions
    
    def _create_action_from_node(self, node: Dict, action_node_map: Dict) -> Optional[Dict]:
        """Create FHIR action from a decision node"""
        action = {
            "id": node['id'],
            "title": node.get('label', node.get('question', 'Clinical Action')),
            "description": node.get('question', ''),
        }
        
        # Add condition expression
        if node.get('condition_expression'):
            action['condition'] = [{
                "kind": "applicability",
                "expression": {
                    "language": "text/cql",
                    "expression": node['condition_expression'],
                    "description": node.get('question')
                }
            }]
        
        # Add timing constraints
        if node.get('timing'):
            action['timingDuration'] = {"value": node['timing']}
        
        # Add priority
        if node.get('priority'):
            action['priority'] = node['priority']
        
        # Add evidence documentation
        if node.get('evidence'):
            evidence = node['evidence']
            action['documentation'] = [{
                "type": "documentation",
                "display": f"Evidence Quality: {evidence.get('quality', 'unknown')}, "
                          f"Recommendation: {evidence.get('recommendation_strength', 'unknown')}"
            }]
        
        # Add related actions from branches
        if node.get('branches'):
            sub_actions = []
            for branch_type, branch_data in node['branches'].items():
                if branch_data.get('actions'):
                    for act in branch_data['actions']:
                        sub_actions.append({
                            "title": f"{branch_type.upper()}: {act}",
                            "description": branch_data.get('description', '')
                        })
            if sub_actions:
                action['action'] = sub_actions
        
        return action
    
    def create_questionnaire(self, extracted_data: Dict, plan_def: PlanDefinition) -> Questionnaire:
        """Create FHIR Questionnaire for clinical assessment"""
        metadata = extracted_data.get('metadata', {})
        questionnaire_id = f"questionnaire-{plan_def.id}"
        
        print(f"📋 Creating Questionnaire: {questionnaire_id}")
        
        items = []
        
        # Add entry criteria questions
        entry_point = extracted_data.get('entry_point', {})
        if entry_point.get('inclusion_criteria'):
            items.append(QuestionnaireItem.construct(
                linkId="inclusion",
                text="Inclusion Criteria",
                type="group",
                item=[
                    QuestionnaireItem.construct(
                        linkId=f"inclusion-{idx}",
                        text=criteria,
                        type="boolean",
                        required=True
                    )
                    for idx, criteria in enumerate(entry_point['inclusion_criteria'])
                ]
            ))
        
        # Add assessment questions
        assessment_questions = extracted_data.get('assessment_questions', [])
        for q in assessment_questions:
            item = QuestionnaireItem.construct(
                linkId=q['id'],
                text=q['text'],
                type=q.get('type', 'string'),
                required=q.get('required', False)
            )
            
            if q.get('options'):
                item.answerOption = [
                    {"valueString": opt} for opt in q['options']
                ]
            
            items.append(item)
        
        # Add questions from decision nodes
        for node in extracted_data.get('decision_nodes', []):
            if node.get('question'):
                items.append(QuestionnaireItem.construct(
                    linkId=node['id'],
                    text=node['question'],
                    type="boolean",
                    required=False
                ))
        
        questionnaire = Questionnaire.construct(
            id=questionnaire_id,
            url=f"{self.base_url}/Questionnaire/{questionnaire_id}",
            name=f"Questionnaire_{plan_def.name}",
            title=f"Clinical Assessment: {plan_def.title}",
            status="draft",
            date=datetime.now().isoformat(),
            publisher=metadata.get('guideline_source', 'Medical Document Converter'),
            description=f"Clinical assessment questionnaire for {plan_def.title}",
            purpose="Structured data collection for clinical decision support",
            item=items if items else [QuestionnaireItem.construct(
                linkId="default",
                text="Clinical assessment required",
                type="display"
            )]
        )
        
        print(f"   ✓ Created {len(items)} questionnaire items")
        return questionnaire
    
    def create_measures(self, extracted_data: Dict, plan_def: PlanDefinition) -> List[Measure]:
        """Create FHIR Measure resources for quality measurement"""
        measures = []
        quality_measures = extracted_data.get('quality_measures', [])
        
        print(f"📊 Creating {len(quality_measures)} Measures...")
        
        for qm in quality_measures:
            measure_id = qm['id']
            
            # Create population groups
            groups = [MeasureGroup.construct(
                id="main-group",
                population=[
                    MeasureGroupPopulation.construct(
                        code=CodeableConcept.construct(
                            coding=[Coding.construct(
                                system="http://terminology.hl7.org/CodeSystem/measure-population",
                                code="numerator",
                                display="Numerator"
                            )]
                        ),
                        description=qm.get('numerator', 'Patients meeting criteria'),
                        criteria={
                            "language": "text/cql",
                            "expression": qm.get('numerator', 'Numerator criteria')
                        }
                    ),
                    MeasureGroupPopulation.construct(
                        code=CodeableConcept.construct(
                            coding=[Coding.construct(
                                system="http://terminology.hl7.org/CodeSystem/measure-population",
                                code="denominator",
                                display="Denominator"
                            )]
                        ),
                        description=qm.get('denominator', 'All eligible patients'),
                        criteria={
                            "language": "text/cql",
                            "expression": qm.get('denominator', 'Denominator criteria')
                        }
                    )
                ]
            )]
            
            measure = Measure.construct(
                id=measure_id,
                url=f"{self.base_url}/Measure/{measure_id}",
                name=self.sanitize_id(qm['title']).replace('-', '_'),
                title=qm['title'],
                status="draft",
                description=qm.get('description'),
                scoring=CodeableConcept.construct(
                    coding=[Coding.construct(
                        system="http://terminology.hl7.org/CodeSystem/measure-scoring",
                        code="proportion",
                        display="Proportion"
                    )]
                ),
                improvementNotation=CodeableConcept.construct(
                    coding=[Coding.construct(
                        system="http://terminology.hl7.org/CodeSystem/measure-improvement-notation",
                        code=qm.get('improvement_notation', 'increase'),
                        display=qm.get('improvement_notation', 'increase').capitalize()
                    )]
                ),
                group=groups
            )
            
            measures.append(measure)
        
        print(f"   ✓ Created {len(measures)} Measures")
        return measures
    
    def create_activity_definitions(self, extracted_data: Dict) -> List[ActivityDefinition]:
        """Create ActivityDefinition resources for clinical actions"""
        action_nodes = extracted_data.get('action_nodes', [])
        activity_defs = []
        
        print(f"🔧 Creating {len(action_nodes)} ActivityDefinitions...")
        
        kind_map = {
            'medication': 'MedicationRequest',
            'procedure': 'Procedure',
            'lab_test': 'ServiceRequest',
            'imaging': 'ServiceRequest',
            'consult': 'ServiceRequest'
        }
        
        for node in action_nodes:
            activity_id = node['id']
            kind = kind_map.get(node.get('type'), 'Task')
            
            activity_def = ActivityDefinition.construct(
                id=activity_id,
                url=f"{self.base_url}/ActivityDefinition/{activity_id}",
                status="draft",
                name=self.sanitize_id(node['title']).replace('-', '_'),
                title=node['title'],
                description=node.get('description'),
                kind=kind,
                intent="proposal",
                priority="routine",
                doNotPerform=False
            )
            
            # Add dosage for medications
            if node.get('type') == 'medication' and node.get('dosage'):
                activity_def.dosage = [{
                    "text": f"{node.get('dosage')} {node.get('route', '')} {node.get('frequency', '')}"
                }]
            
            activity_defs.append(activity_def)
        
        print(f"   ✓ Created {len(activity_defs)} ActivityDefinitions")
        return activity_defs
    
    def create_evidence_resources(self, extracted_data: Dict, plan_def: PlanDefinition) -> List[Evidence]:
        """Create Evidence resources for clinical recommendations"""
        evidence_resources = []
        
        print(f"🔬 Creating Evidence resources...")
        
        for idx, node in enumerate(extracted_data.get('decision_nodes', [])):
            if node.get('evidence'):
                evidence_id = f"evidence-{node['id']}"
                evidence_data = node['evidence']
                
                evidence = Evidence.construct(
                    id=evidence_id,
                    url=f"{self.base_url}/Evidence/{evidence_id}",
                    status="draft",
                    title=f"Evidence for {node.get('label', f'Decision {idx}')}",
                    description=node.get('question', 'Clinical evidence'),
                    certainty=[{
                        "rating": CodeableConcept.construct(
                            text=evidence_data.get('quality', 'moderate')
                        ),
                        "note": [{
                            "text": f"Recommendation strength: {evidence_data.get('recommendation_strength', 'weak')}"
                        }]
                    }]
                )
                
                evidence_resources.append(evidence)
        
        print(f"   ✓ Created {len(evidence_resources)} Evidence resources")
        return evidence_resources
    
    def create_evidence_variables(self, extracted_data: Dict) -> List[EvidenceVariable]:
        """Create EvidenceVariable resources for measurements"""
        evidence_variables = []
        
        print(f"📈 Creating EvidenceVariable resources...")
        
        for node in extracted_data.get('decision_nodes', []):
            if node.get('measurement'):
                measurement = node['measurement']
                var_id = f"var-{node['id']}"
                
                evidence_var = EvidenceVariable.construct(
                    id=var_id,
                    url=f"{self.base_url}/EvidenceVariable/{var_id}",
                    status="draft",
                    name=self.sanitize_id(measurement.get('parameter', 'variable')).replace('-', '_'),
                    title=measurement.get('parameter', 'Clinical Variable'),
                    description=f"Measurement: {measurement.get('parameter')} "
                               f"(Unit: {measurement.get('unit', 'N/A')})",
                    note=[{
                        "text": f"Normal range: {measurement.get('normal_range', 'N/A')}, "
                               f"Threshold: {measurement.get('threshold', 'N/A')}"
                    }]
                )
                
                evidence_variables.append(evidence_var)
        
        print(f"   ✓ Created {len(evidence_variables)} EvidenceVariable resources")
        return evidence_variables
    
    def create_measure_report(self, measure: Measure, plan_def: PlanDefinition) -> MeasureReport:
        """Create template MeasureReport"""
        report_id = f"report-{measure.id}"
        
        measure_report = MeasureReport.construct(
            id=report_id,
            status="complete",
            type="summary",
            measure=f"{self.base_url}/Measure/{measure.id}",
            date=datetime.now().isoformat(),
            period={
                "start": datetime.now().isoformat(),
                "end": datetime.now().isoformat()
            },
            group=[{
                "id": "main-group",
                "population": [
                    {
                        "code": CodeableConcept.construct(
                            coding=[Coding.construct(
                                system="http://terminology.hl7.org/CodeSystem/measure-population",
                                code="numerator"
                            )]
                        ),
                        "count": 0
                    },
                    {
                        "code": CodeableConcept.construct(
                            coding=[Coding.construct(
                                system="http://terminology.hl7.org/CodeSystem/measure-population",
                                code="denominator"
                            )]
                        ),
                        "count": 0
                    }
                ]
            }]
        )
        
        return measure_report
    
    def create_library(self, plan_def: PlanDefinition, 
                      arden_content: Optional[str] = None,
                      cql_content: Optional[str] = None) -> Library:
        """Create Library resource with clinical logic"""
        
        library_id = f"library-{plan_def.id}"
        
        print(f"📚 Creating Library: {library_id}")
        
        library = Library.construct(
            id=library_id,
            url=f"{self.base_url}/Library/{library_id}",
            status="draft",
            type=CodeableConcept.construct(
                coding=[Coding.construct(
                    system="http://terminology.hl7.org/CodeSystem/library-type",
                    code="logic-library",
                    display="Logic Library"
                )]
            ),
            name=f"Library_{plan_def.name}",
            title=f"Clinical Logic for {plan_def.title}",
            description="Contains clinical decision logic extracted from medical document",
            purpose="Provide executable logic for clinical decision support"
        )
        
        # Add content
        content = []
        if cql_content:
            content.append({
                "contentType": "text/cql",
                "data": base64.b64encode(cql_content.encode()).decode()
            })
        if arden_content:
            content.append({
                "contentType": "text/plain",
                "title": "Arden Syntax MLM",
                "data": base64.b64encode(arden_content.encode()).decode()
            })
        
        if content:
            library.content = content
            print(f"   ✓ Added {len(content)} logic content items")
        
        return library
    
    def create_bundle(self, resources: Dict) -> Bundle:
        """Create comprehensive FHIR Bundle"""
        print("📦 Creating FHIR Bundle...")
        
        entries = []
        
        # Add all resources to bundle
        resource_types = [
            ('plan_definition', 'PlanDefinition'),
            ('questionnaire', 'Questionnaire'),
            ('library', 'Library'),
            ('activity_definitions', 'ActivityDefinition'),
            ('measures', 'Measure'),
            ('measure_reports', 'MeasureReport'),
            ('evidence', 'Evidence'),
            ('evidence_variables', 'EvidenceVariable')
        ]
        
        for key, resource_type in resource_types:
            resource_list = resources.get(key, [])
            if not isinstance(resource_list, list):
                resource_list = [resource_list]
            
            for resource in resource_list:
                if resource:
                    entries.append(BundleEntry.construct(
                        fullUrl=f"{self.base_url}/{resource_type}/{resource.id}",
                        resource=resource
                    ))
        
        bundle = Bundle.construct(
            id=f"bundle-{resources['plan_definition'].id}",
            type="collection",
            timestamp=datetime.now().isoformat(),
            entry=entries
        )
        
        print(f"   ✓ Bundle created with {len(entries)} entries")
        return bundle


class CPGBuilder:
    """Main class for building comprehensive CPG from medical documents"""
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None):
        self.extractor = MedicalDocumentExtractor(llm_provider, api_key)
        self.converter = FHIRCPGConverter()
        self.arden_generator = ArdenSyntaxGenerator()
        self.cql_generator = CQLGenerator(self.extractor.client)
    
    def build_cpg(self, document_path: str, base_output_dir: str = "cpg_output", 
                  translate_to_vietnamese: bool = True) -> Dict:
        """
        Build comprehensive CPG from medical document
        
        Args:
            document_path: Path to medical document (PDF, DOC, DOCX, TXT)
            base_output_dir: Base directory for all CPG outputs
            translate_to_vietnamese: Translate English documents to Vietnamese
            
        Returns:
            Dictionary with all generated resources
        """
        print("\n" + "="*80)
        print("🏥 COMPREHENSIVE CPG BUILDER WITH VIETNAMESE SUPPORT")
        print("="*80 + "\n")
        
        # Step 1: Extract data
        print("STEP 1: Extracting clinical logic from document")
        print("-" * 80)
        extracted_data = self.extractor.extract_from_document(document_path, translate_to_vietnamese)
        
        # Create disease-specific folder
        disease = extracted_data.get('metadata', {}).get('disease', 'unknown_condition')
        disease_folder = self.converter.sanitize_id(disease)
        output_dir = Path(base_output_dir) / disease_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save extracted data
        with open(output_dir / "extracted_data.json", 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {output_dir}/extracted_data.json\n")
        
        # Step 2: Create FHIR resources
        print("STEP 2: Creating FHIR CPG resources")
        print("-" * 80)
        
        plan_def = self.converter.create_plan_definition(extracted_data)
        
        # Generate Arden Syntax first
        print("\nSTEP 3: Generating Arden Syntax MLM")
        print("-" * 80)
        arden_mlm = self.arden_generator.generate(extracted_data, plan_def)
        
        # Save Arden Syntax immediately
        with open(output_dir / "arden_syntax.mlm", 'w', encoding='utf-8') as f:
            f.write(arden_mlm)
        print(f"💾 Saved: arden_syntax.mlm")
        
        # Generate CQL from Arden Syntax
        print("\nSTEP 4: Generating CQL from Arden Syntax")
        print("-" * 80)
        cql_content = self.cql_generator.generate_from_arden(arden_mlm, extracted_data, plan_def)
        
        # Save CQL
        with open(output_dir / "logic.cql", 'w', encoding='utf-8') as f:
            f.write(cql_content)
        print(f"💾 Saved: logic.cql")
        
        # Continue with other resources
        print("\nSTEP 5: Creating remaining FHIR resources")
        print("-" * 80)
        
        questionnaire = self.converter.create_questionnaire(extracted_data, plan_def)
        activity_defs = self.converter.create_activity_definitions(extracted_data)
        measures = self.converter.create_measures(extracted_data, plan_def)
        evidence = self.converter.create_evidence_resources(extracted_data, plan_def)
        evidence_vars = self.converter.create_evidence_variables(extracted_data)
        measure_reports = [self.converter.create_measure_report(m, plan_def) for m in measures]
        library = self.converter.create_library(plan_def, 
                                                arden_content=arden_mlm,
                                                cql_content=cql_content)
        
        resources = {
            'plan_definition': plan_def,
            'questionnaire': questionnaire,
            'activity_definitions': activity_defs,
            'measures': measures,
            'measure_reports': measure_reports,
            'evidence': evidence,
            'evidence_variables': evidence_vars,
            'library': library
        }
        
        bundle = self.converter.create_bundle(resources)
        
        # Step 6: Save all resources
        print("\nSTEP 6: Saving FHIR resources")
        print("-" * 80)
        
        self._save_resources(resources, bundle, output_dir, extracted_data)
        
        # Step 7: Generate summary
        self._print_summary(resources, output_dir, disease, extracted_data)
        
        return {
            'extracted_data': extracted_data,
            'output_directory': str(output_dir),
            'disease': disease,
            **resources,
            'bundle': bundle
        }
    
    def _save_resources(self, resources: Dict, bundle: Bundle, 
                       output_dir: Path, extracted_data: Dict):
        """Save all FHIR resources to files"""
        
        # Save bundle
        with open(output_dir / "bundle.json", 'w', encoding='utf-8') as f:
            f.write(bundle.json(indent=2))
        print(f"💾 Bundle: bundle.json")
        
        # Save PlanDefinition
        with open(output_dir / "plandefinition.json", 'w', encoding='utf-8') as f:
            f.write(resources['plan_definition'].json(indent=2))
        print(f"💾 PlanDefinition: plandefinition.json")
        
        # Save Questionnaire
        with open(output_dir / "questionnaire.json", 'w', encoding='utf-8') as f:
            f.write(resources['questionnaire'].json(indent=2))
        print(f"💾 Questionnaire: questionnaire.json")
        
        # Save Library
        with open(output_dir / "library.json", 'w', encoding='utf-8') as f:
            f.write(resources['library'].json(indent=2))
        print(f"💾 Library: library.json")
        
        # Save ActivityDefinitions
        if resources['activity_definitions']:
            activity_dir = output_dir / "activities"
            activity_dir.mkdir(exist_ok=True)
            for activity in resources['activity_definitions']:
                with open(activity_dir / f"{activity.id}.json", 'w', encoding='utf-8') as f:
                    f.write(activity.json(indent=2))
            print(f"💾 ActivityDefinitions: {len(resources['activity_definitions'])} files in activities/")
        
        # Save Measures
        if resources['measures']:
            measure_dir = output_dir / "measures"
            measure_dir.mkdir(exist_ok=True)
            for measure in resources['measures']:
                with open(measure_dir / f"{measure.id}.json", 'w', encoding='utf-8') as f:
                    f.write(measure.json(indent=2))
            print(f"💾 Measures: {len(resources['measures'])} files in measures/")
        
        # Save MeasureReports
        if resources['measure_reports']:
            report_dir = output_dir / "measure_reports"
            report_dir.mkdir(exist_ok=True)
            for report in resources['measure_reports']:
                with open(report_dir / f"{report.id}.json", 'w', encoding='utf-8') as f:
                    f.write(report.json(indent=2))
            print(f"💾 MeasureReports: {len(resources['measure_reports'])} files in measure_reports/")
        
        # Save Evidence
        if resources['evidence']:
            evidence_dir = output_dir / "evidence"
            evidence_dir.mkdir(exist_ok=True)
            for ev in resources['evidence']:
                with open(evidence_dir / f"{ev.id}.json", 'w', encoding='utf-8') as f:
                    f.write(ev.json(indent=2))
            print(f"💾 Evidence: {len(resources['evidence'])} files in evidence/")
        
        # Save EvidenceVariables
        if resources['evidence_variables']:
            var_dir = output_dir / "evidence_variables"
            var_dir.mkdir(exist_ok=True)
            for var in resources['evidence_variables']:
                with open(var_dir / f"{var.id}.json", 'w', encoding='utf-8') as f:
                    f.write(var.json(indent=2))
            print(f"💾 EvidenceVariables: {len(resources['evidence_variables'])} files in evidence_variables/")
        
        # Save README
        readme = self._generate_readme(extracted_data, resources)
        with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme)
        print(f"💾 Documentation: README.md")
    
    def _generate_readme(self, extracted_data: Dict, resources: Dict) -> str:
        """Generate README documentation"""
        metadata = extracted_data.get('metadata', {})
        language_info = extracted_data.get('language', {})
        
        readme = f"""# Clinical Practice Guideline: {metadata.get('title', 'N/A')}

## Overview

**Disease/Condition:** {metadata.get('disease', 'N/A')}  
**Medical Domain:** {metadata.get('domain', 'N/A')}  
**Target Population:** {metadata.get('target_population', 'N/A')}  
**Evidence Level:** {metadata.get('evidence_level', 'N/A')}  
**Source:** {metadata.get('guideline_source', 'N/A')}  
**Language:** {'Vietnamese (Translated)' if language_info.get('translated') else 'Vietnamese' if language_info.get('detected') == 'vi' else 'English'}

## Purpose

{metadata.get('purpose', 'N/A')}

## Directory Structure

```
.
├── bundle.json                  # Complete FHIR Bundle
├── plandefinition.json         # Main clinical protocol
├── questionnaire.json          # Clinical assessment form
├── library.json                # Logic library reference
├── logic.cql                   # CQL decision logic (generated from Arden)
├── arden_syntax.mlm            # Arden Syntax MLM (source)
├── activities/                 # Clinical actions
│   └── *.json
├── measures/                   # Quality measures
│   └── *.json
├── measure_reports/            # Measurement templates
│   └── *.json
├── evidence/                   # Clinical evidence
│   └── *.json
├── evidence_variables/         # Measurement variables
│   └── *.json
├── extracted_data.json         # Raw extracted data
└── README.md                   # This file
```

## FHIR Resources

### Core Resources
- **PlanDefinition**: {resources['plan_definition'].id}
  - {len(resources['plan_definition'].action or [])} clinical actions defined
  
- **Questionnaire**: {resources['questionnaire'].id}
  - {len(resources['questionnaire'].item or [])} assessment items

- **Library**: {resources['library'].id}
  - Contains executable clinical logic in CQL and Arden Syntax

### Supporting Resources
- **ActivityDefinitions**: {len(resources['activity_definitions'])} actions
- **Measures**: {len(resources['measures'])} quality measures
- **Evidence**: {len(resources['evidence'])} evidence items
- **EvidenceVariables**: {len(resources['evidence_variables'])} measurement variables
- **MeasureReports**: {len(resources['measure_reports'])} report templates

## Clinical Workflow

### Entry Criteria
"""
        
        entry = extracted_data.get('entry_point', {})
        if entry.get('inclusion_criteria'):
            readme += "\n**Inclusion Criteria:**\n"
            for criteria in entry['inclusion_criteria']:
                readme += f"- {criteria}\n"
        
        if entry.get('exclusion_criteria'):
            readme += "\n**Exclusion Criteria:**\n"
            for criteria in entry['exclusion_criteria']:
                readme += f"- {criteria}\n"
        
        readme += f"""

### Decision Points

{len(extracted_data.get('decision_nodes', []))} clinical decision points defined in the guideline.

### Clinical Actions

{len(extracted_data.get('action_nodes', []))} specific clinical actions defined.

## Logic Implementation

### Arden Syntax MLM
The `arden_syntax.mlm` file contains the Medical Logic Module in Arden Syntax format. This is the primary source for clinical decision logic extracted from the guideline document.

### CQL Generation
The `logic.cql` file is automatically generated from the Arden Syntax MLM using AI-powered conversion. This provides:
- FHIR 4.0.1 compatibility
- Proper value sets and codes
- Reusable definitions
- Integration with FHIRHelpers

## Quality Measures
"""
        
        for measure in extracted_data.get('quality_measures', []):
            readme += f"""
### {measure['title']}
- **Type**: {measure.get('type', 'N/A')}
- **Description**: {measure.get('description', 'N/A')}
- **Improvement**: {measure.get('improvement_notation', 'increase')}
"""
        
        readme += """

## Implementation Guide

### Using the PlanDefinition
1. Load the bundle.json into your FHIR server
2. Reference the PlanDefinition in your CDS hooks
3. Use the Questionnaire for data collection
4. Apply the decision logic from logic.cql

### Logic Execution
1. **Arden Syntax**: Use in Arden-compatible clinical systems
2. **CQL**: Implement in CQL execution engines for FHIR CDS
3. **Workflow**: Arden MLM → CQL → FHIR CDS Integration

### Measuring Quality
1. Implement the Measure definitions
2. Generate MeasureReports for your population
3. Track adherence to the clinical guideline
4. Monitor outcomes over time

### Evidence Integration
- Review Evidence resources for supporting literature
- Check evidence quality ratings
- Validate recommendations against local protocols

## Integration Points

### EHR Integration
- Import FHIR Bundle to EHR
- Map questionnaire to intake forms
- Configure CDS alerts based on PlanDefinition

### Clinical Decision Support
- Implement CQL logic in CDS engine
- Trigger alerts at appropriate decision points
- Display recommendations to clinicians

### Quality Reporting
- Generate MeasureReports regularly
- Submit to quality registries
- Track compliance metrics

## Vietnamese Language Support

This CPG has been processed with Vietnamese language support:
- Original text extracted from document
- Translation to Vietnamese (if source was English)
- Clinical terminology preserved
- All FHIR resources support UTF-8 encoding

## Next Steps

1. **Validation**: Review extracted logic with clinical experts
2. **Testing**: Test decision logic with sample cases
3. **Integration**: Configure in your FHIR server/CDS system
4. **Monitoring**: Track measure reports and outcomes
5. **Optimization**: Refine CQL based on Arden Syntax feedback

## Generated

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Generator**: CPG Builder with Vietnamese Support  
**FHIR Version**: 4.0.1  
**Logic Format**: Arden Syntax → CQL

## Support

For questions or issues with this CPG implementation, please review:
- FHIR PlanDefinition documentation
- CQL specification
- Arden Syntax documentation
- Your organization's clinical informatics team
"""
        
        return readme
    
    def _print_summary(self, resources: Dict, output_dir: Path, disease: str, extracted_data: Dict):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("✅ CPG BUILD COMPLETE")
        print("="*80)
        
        language_info = extracted_data.get('language', {})
        
        print(f"\n🏥 Disease/Condition: {disease.replace('-', ' ').title()}")
        print(f"🌐 Language: {'Vietnamese (Translated)' if language_info.get('translated') else 'Vietnamese' if language_info.get('detected') == 'vi' else 'English'}")
        print(f"📁 Output Directory: {output_dir}/")
        
        print(f"\n📊 FHIR Resources Generated:")
        print(f"   ✓ PlanDefinition: 1 (with {len(resources['plan_definition'].action or [])} actions)")
        print(f"   ✓ Questionnaire: 1 (with {len(resources['questionnaire'].item or [])} items)")
        print(f"   ✓ Library: 1 (with CQL and Arden Syntax logic)")
        print(f"   ✓ ActivityDefinitions: {len(resources['activity_definitions'])}")
        print(f"   ✓ Measures: {len(resources['measures'])}")
        print(f"   ✓ MeasureReports: {len(resources['measure_reports'])}")
        print(f"   ✓ Evidence: {len(resources['evidence'])}")
        print(f"   ✓ EvidenceVariables: {len(resources['evidence_variables'])}")
        print(f"   ✓ Bundle: 1 (complete collection)")
        
        total_resources = (1 + 1 + 1 + len(resources['activity_definitions']) + 
                          len(resources['measures']) + len(resources['measure_reports']) +
                          len(resources['evidence']) + len(resources['evidence_variables']))
        
        print(f"\n   📦 Total Resources: {total_resources}")
        
        print(f"\n📝 Additional Files:")
        print(f"   ✓ arden_syntax.mlm - Arden Syntax MLM (SOURCE)")
        print(f"   ✓ logic.cql - CQL decision logic (Generated from Arden)")
        print(f"   ✓ README.md - Implementation guide")
        print(f"   ✓ extracted_data.json - Raw extraction data")
        
        print(f"\n🔄 Logic Workflow:")
        print(f"   Medical Document → Arden Syntax MLM → CQL → FHIR Resources")
        
        print(f"\n🎯 Use Cases:")
        print(f"   • Import bundle into FHIR-compliant EHR")
        print(f"   • Implement clinical decision support")
        print(f"   • Quality measurement and reporting")
        print(f"   • Clinical pathway optimization")
        print(f"   • Evidence-based practice validation")
        print(f"   • Vietnamese healthcare system integration")
        
        print(f"\n📚 Next Steps:")
        print(f"   1. Review extracted data for accuracy")
        print(f"   2. Validate Arden Syntax MLM with clinical experts")
        print(f"   3. Test CQL logic generated from Arden")
        print(f"   4. Configure measures for your population")
        print(f"   5. Deploy to FHIR server and CDS system")
        
        print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build comprehensive CPG from medical document with Vietnamese support"
    )
    parser.add_argument(
        "document_path",
        help="Path to medical document (PDF, DOC, DOCX, TXT)"
    )
    parser.add_argument(
        "--output-dir",
        default="cpg_output",
        help="Base output directory (default: cpg_output)"
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Skip Vietnamese translation (keep original language)"
    )
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai"],
        help="LLM provider for extraction (default: openai)"
    )
    parser.add_argument(
        "--api-key",
        help="API key (uses env variable if not provided)"
    )
    
    args = parser.parse_args()
    
    try:
        builder = CPGBuilder(
            llm_provider=args.llm_provider,
            api_key=args.api_key
        )
        
        result = builder.build_cpg(
            document_path=args.document_path,
            base_output_dir=args.output_dir,
            translate_to_vietnamese=not args.no_translate
        )
        
        print(f"✨ Success! CPG built for: {result['disease']}")
        print(f"   Output: {result['output_directory']}")
        print(f"\n💡 Key Features:")
        print(f"   • Arden Syntax MLM saved first for validation")
        print(f"   • CQL generated from validated Arden Syntax")
        print(f"   • Vietnamese language support enabled")
        print(f"   • Complete FHIR CPG resource bundle")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)