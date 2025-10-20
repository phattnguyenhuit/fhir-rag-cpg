from fhir.resources.plandefinition import PlanDefinition
from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.bundle import Bundle, BundleEntry
from fhir.resources.library import Library
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from datetime import datetime
import re
import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAPI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI not available. Install: pip install openai")

class MedicalDiagramExtractor:
    """Extract structured medical data from diagrams using LLM Vision APIs"""
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize extractor with LLM provider
        
        Args:
            llm_provider: "openai" (GPT-4 Vision) or "anthropic" (Claude Vision)
            api_key: API key or None to use environment variable
        """
        self.llm_provider = llm_provider
        self.client = None
        
        if llm_provider == "openai" and HAS_OPENAI:
            self.client = OpenAI(api_key= OPENAPI_API_KEY or os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported LLM provider or missing library: {llm_provider}")
    
    def encode_image(self, image_path: str) -> tuple[str, str]:
        """Encode image to base64 and determine media type"""
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        
        ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/png')
        
        return encoded, media_type
       
    def create_extraction_prompt(self) -> str:
        """Create detailed prompt for medical diagram extraction"""
        return """Analyze this medical flowchart/diagram carefully and extract ALL clinical decision logic.

        You are analyzing a clinical decision support diagram (like stroke diagnosis, fever workup, etc.).

        Return a JSON object with this EXACT structure:

        {
        "metadata": {
            "title": "Full title of the clinical algorithm",
            "domain": "Medical domain (e.g., Neurology, Infectious Disease)",
            "purpose": "Clinical purpose of this algorithm",
            "target_population": "Target patient population"
        },
        "entry_point": {
            "id": "entry",
            "condition": "Initial presenting condition/symptom",
            "description": "Entry criteria description"
        },
        "decision_nodes": [
            {
            "id": "unique_node_id",
            "type": "condition|action|assessment|classification",
            "label": "The text label on this node",
            "question": "The decision question being asked",
            "condition_expression": "Logical condition (e.g., 'systolic BP > 140')",
            "branches": {
                "yes": {
                "next_node": "id_of_next_node",
                "actions": ["action1", "action2"],
                "description": "What happens if yes"
                },
                "no": {
                "next_node": "id_of_next_node",
                "actions": ["action1"],
                "description": "What happens if no"
                }
            },
            "timing": "Any timing constraints (e.g., 'within 4.5 hours')",
            "priority": "urgent|routine|stat",
            "clinical_codes": {
                "snomed": [],
                "icd10": [],
                "loinc": []
            }
            }
        ],
        "action_nodes": [
            {
            "id": "unique_action_id",
            "type": "medication|procedure|lab_test|imaging|consult",
            "title": "Action title",
            "description": "Detailed description",
            "instructions": "Step-by-step instructions",
            "contraindications": ["contraindication1"],
            "precautions": ["precaution1"]
            }
        ],
        "endpoints": [
            {
            "id": "endpoint_id",
            "category": "diagnosis|treatment|referral",
            "diagnosis": "Final diagnosis",
            "recommendations": ["recommendation1", "recommendation2"],
            "follow_up": "Follow-up instructions"
            }
        ],
        "metadata_extracted": {
            "total_decision_points": 0,
            "max_depth": 0,
            "critical_paths": []
        }
        }

        IMPORTANT INSTRUCTIONS:
        1. Extract EVERY text label, condition, and branch in the diagram
        2. Preserve the exact decision logic flow
        3. Identify temporal constraints (time windows, sequences)
        4. Note any measurements with units (e.g., "BP > 180/110 mmHg")
        5. Capture all clinical actions and recommendations
        6. If you see medical codes (SNOMED, ICD-10, LOINC), include them
        7. Be precise with medical terminology
        8. Maintain the hierarchical structure of decisions

        Return ONLY valid JSON, no markdown formatting."""
    
    def extract_from_image(self, image_path: str) -> Dict:
        """
        Extract structured clinical data from medical diagram
        
        Args:
            image_path: Path to the medical diagram image
            
        Returns:
            Dictionary containing structured clinical logic
        """
        if not self.client:
            raise ValueError(f"LLM client not initialized for {self.llm_provider}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"📊 Analyzing medical diagram: {image_path}")
        
        base64_image, media_type = self.encode_image(image_path)
        extraction_prompt = self.create_extraction_prompt()
        
        try:
            if self.llm_provider == "openai":
                print("🤖 Using GPT-4 Vision for extraction...")
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": extraction_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.1
                )
                content = response.choices[0].message.content
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
            # Parse JSON response (handle code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            extracted_data = json.loads(content)
            
            print(f"✅ Extraction complete!")
            print(f"   - Decision nodes: {len(extracted_data.get('decision_nodes', []))}")
            print(f"   - Action nodes: {len(extracted_data.get('action_nodes', []))}")
            print(f"   - Endpoints: {len(extracted_data.get('endpoints', []))}")
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw response: {content[:500]}...")
            raise
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            raise
class FHIRConverter:
    """Convert extracted medical data to FHIR-compliant resources"""
    
    def __init__(self):
        self.base_url = "http://example.org/fhir"
    
    def create_plan_definition(self, extracted_data: Dict) -> PlanDefinition:
        """
        Create FHIR PlanDefinition from extracted diagram data
        
        Args:
            extracted_data: Structured data from diagram extraction
            
        Returns:
            FHIR PlanDefinition resource
        """
        metadata = extracted_data.get('metadata', {})
        title = metadata.get('title', 'Clinical Decision Algorithm')
        plan_id = re.sub(r'[^a-zA-Z0-9-]', '-', title.lower())
        
        print(f"📝 Creating PlanDefinition: {title}")
        
        # Build actions from decision nodes
        actions = self._build_actions_from_nodes(
            extracted_data.get('decision_nodes', []),
            extracted_data.get('action_nodes', [])
        )
        
        # Create PlanDefinition
        plan_def = PlanDefinition.construct(
            id=plan_id,
            url=f"{self.base_url}/PlanDefinition/{plan_id}",
            identifier=[{
                "system": f"{self.base_url}/identifiers",
                "value": plan_id
            }],
            version="1.0.0",
            name=title.replace(' ', '_'),
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
            publisher="Medical Diagram Converter",
            description=metadata.get('purpose', 'Clinical decision support protocol'),
            purpose=metadata.get('purpose'),
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
                    text=metadata.get('domain', 'Clinical Decision Support')
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
    
    def _create_action_from_node(self, node: Dict, 
                                 action_node_map: Dict) -> Optional[Dict]:
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
            action['timingDuration'] = {
                "value": node['timing']
            }
        
        # Add priority
        if node.get('priority'):
            action['priority'] = node['priority']
        
        # Add related actions from branches
        if node.get('branches'):
            related_actions = []
            sub_actions = []
            
            for branch_type, branch_data in node['branches'].items():
                if branch_data.get('next_node'):
                    related_actions.append({
                        "actionId": branch_data['next_node'],
                        "relationship": "before-start",
                        "offsetDuration": {
                            "value": 0,
                            "unit": "min"
                        }
                    })
                
                # Create sub-actions for branch actions
                if branch_data.get('actions'):
                    for act in branch_data['actions']:
                        sub_actions.append({
                            "title": f"{branch_type.upper()}: {act}",
                            "description": branch_data.get('description', '')
                        })
            
            if related_actions:
                action['relatedAction'] = related_actions
            if sub_actions:
                action['action'] = sub_actions
        
        # Add documentation for clinical codes
        if node.get('clinical_codes'):
            docs = []
            for code_system, codes in node['clinical_codes'].items():
                if codes:
                    docs.append({
                        "type": "documentation",
                        "display": f"{code_system.upper()}: {', '.join(codes)}"
                    })
            if docs:
                action['documentation'] = docs
        
        return action
    
    def create_activity_definitions(self, extracted_data: Dict) -> List[ActivityDefinition]:
        """Create ActivityDefinition resources for clinical actions"""
        
        action_nodes = extracted_data.get('action_nodes', [])
        activity_defs = []
        
        print(f"📋 Creating {len(action_nodes)} ActivityDefinitions...")
        
        for node in action_nodes:
            activity_id = node['id']
            
            # Map action type to FHIR kind
            kind_map = {
                'medication': 'MedicationRequest',
                'procedure': 'Procedure',
                'lab_test': 'ServiceRequest',
                'imaging': 'ServiceRequest',
                'consult': 'ServiceRequest'
            }
            kind = kind_map.get(node.get('type'), 'Task')
            
            activity_def = ActivityDefinition.construct(
                id=activity_id,
                url=f"{self.base_url}/ActivityDefinition/{activity_id}",
                status="draft",
                name=node['title'].replace(' ', '_'),
                title=node['title'],
                description=node.get('description'),
                kind=kind,
                intent="proposal",
                priority="routine",
                doNotPerform=False
            )
            
            activity_defs.append(activity_def)
        
        print(f"   ✓ Created {len(activity_defs)} ActivityDefinitions")
        return activity_defs
    
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
            description="Contains clinical decision logic extracted from medical diagram",
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
    
    def create_bundle(self, plan_def: PlanDefinition,
                     activity_defs: List[ActivityDefinition],
                     library: Library) -> Bundle:
        """Create FHIR Bundle containing all resources"""
        
        print("📦 Creating FHIR Bundle...")
        
        entries = []
        
        # Add PlanDefinition
        entries.append(BundleEntry.construct(
            fullUrl=f"{self.base_url}/PlanDefinition/{plan_def.id}",
            resource=plan_def
        ))
        
        # Add ActivityDefinitions
        for activity_def in activity_defs:
            entries.append(BundleEntry.construct(
                fullUrl=f"{self.base_url}/ActivityDefinition/{activity_def.id}",
                resource=activity_def
            ))
        
        # Add Library
        entries.append(BundleEntry.construct(
            fullUrl=f"{self.base_url}/Library/{library.id}",
            resource=library
        ))
        
        # Create Bundle
        bundle = Bundle.construct(
            id=f"bundle-{plan_def.id}",
            type="collection",
            timestamp=datetime.now().isoformat(),
            entry=entries
        )
        
        print(f"   ✓ Bundle created with {len(entries)} entries")
        return bundle


class ArdenSyntaxGenerator:
    """Generate Arden Syntax MLM from FHIR resources"""
    
    def generate(self, plan_def: PlanDefinition, extracted_data: Dict) -> str:
        """Generate Arden Syntax Medical Logic Module"""
        
        print("⚕️  Generating Arden Syntax MLM...")
        
        title = plan_def.title.replace(" ", "_")
        
        mlm = f"""maintenance:
    title: {title};;
    mlmname: {plan_def.id};;
    arden: Version 2.10;;
    version: {plan_def.version};;
    institution: Generated from Medical Diagram;;
    author: Automated Conversion System;;
    specialist: ;;
    date: {datetime.now().strftime('%Y-%m-%d')};;
    validation: testing;;

library:
    purpose: {plan_def.description};;
    explanation: This MLM implements clinical decision logic extracted from
                 a medical flowchart/diagram using AI vision technology and
                 converted to FHIR PlanDefinition format.;;
    keywords: {', '.join([topic.text for topic in plan_def.topic if topic.text])};;
    citations: ;;

knowledge:
    type: data-driven;;
    
    data:
        /* Patient identifiers */
        patient_id := read {{patient_identifier}};
        encounter_id := read {{encounter_id}};
        
        /* Clinical data elements from diagram */
"""
        
        # Add data elements from decision nodes
        decision_nodes = extracted_data.get('decision_nodes', [])
        for node in decision_nodes:
            if node.get('condition_expression'):
                var_name = re.sub(r'[^a-zA-Z0-9_]', '_', node['id'])
                mlm += f"        {var_name} := read {{{node['condition_expression']}}};\n"
        
        mlm += """    ;;
    
    evoke:
        /* Trigger conditions */
"""
        
        # Add evoke from entry point
        entry_point = extracted_data.get('entry_point', {})
        if entry_point.get('condition'):
            mlm += f"        {entry_point['condition']} OR\n"
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
            mlm += f"\n        /* {node.get('label', f'Decision {idx}')} */\n"
            
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


class MedicalDiagramToFHIR:
    """Main class combining extraction and FHIR conversion"""
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None):
        self.extractor = MedicalDiagramExtractor(llm_provider, api_key)
        self.converter = FHIRConverter()
        self.arden_generator = ArdenSyntaxGenerator()
    
    def process_diagram(self, image_path: str, output_dir: str = "output") -> Dict:
        """
        Complete pipeline: Extract from diagram and convert to FHIR
        
        Args:
            image_path: Path to medical diagram image
            output_dir: Directory for output files
            
        Returns:
            Dictionary with all generated resources
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        print("\n" + "="*70)
        print(f"🏥 MEDICAL DIAGRAM TO FHIR CONVERTER")
        print("="*70 + "\n")
        
        # Step 1: Extract structured data from diagram
        print("STEP 1: Extracting clinical logic from diagram")
        print("-" * 70)
        extracted_data = self.extractor.extract_from_image(image_path)
        
        # Save extracted data
        extracted_path = os.path.join(output_dir, f"{base_name}_extracted.json")
        with open(extracted_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2)
        print(f"💾 Saved: {extracted_path}\n")
        
        # Step 2: Convert to FHIR resources
        print("STEP 2: Converting to FHIR-compliant resources")
        print("-" * 70)
        
        plan_def = self.converter.create_plan_definition(extracted_data)
        activity_defs = self.converter.create_activity_definitions(extracted_data)
        
        # Generate Arden Syntax
        arden_mlm = self.arden_generator.generate(plan_def, extracted_data)
        
        # Create Library with Arden content
        library = self.converter.create_library(plan_def, arden_content=arden_mlm)
        
        # Create Bundle
        bundle = self.converter.create_bundle(plan_def, activity_defs, library)
        
        # Step 3: Save all outputs
        print("\n" + "STEP 3: Saving FHIR resources")
        print("-" * 70)
        
        # Save Bundle
        bundle_path = os.path.join(output_dir, f"{base_name}_bundle.json")
        with open(bundle_path, 'w', encoding='utf-8') as f:
            f.write(bundle.json(indent=2))
        print(f"💾 Bundle: {bundle_path}")
        
        # Save PlanDefinition
        plan_path = os.path.join(output_dir, f"{base_name}_plandefinition.json")
        with open(plan_path, 'w', encoding='utf-8') as f:
            f.write(plan_def.json(indent=2))
        print(f"💾 PlanDefinition: {plan_path}")
        
        # Save ActivityDefinitions
        for idx, activity_def in enumerate(activity_defs):
            activity_path = os.path.join(output_dir, 
                                        f"{base_name}_activity_{activity_def.id}.json")
            with open(activity_path, 'w', encoding='utf-8') as f:
                f.write(activity_def.json(indent=2))
        print(f"💾 ActivityDefinitions: {len(activity_defs)} files")
        
        # Save Library
        library_path = os.path.join(output_dir, f"{base_name}_library.json")
        with open(library_path, 'w', encoding='utf-8') as f:
            f.write(library.json(indent=2))
        print(f"💾 Library: {library_path}")
        
        # Save Arden MLM
        arden_path = os.path.join(output_dir, f"{base_name}.mlm")
        with open(arden_path, 'w', encoding='utf-8') as f:
            f.write(arden_mlm)
        print(f"💾 Arden Syntax: {arden_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("✅ CONVERSION COMPLETE")
        print("="*70)
        print(f"\n📊 Statistics:")
        print(f"   • Decision nodes: {len(extracted_data.get('decision_nodes', []))}")
        print(f"   • Action nodes: {len(extracted_data.get('action_nodes', []))}")
        print(f"   • FHIR resources: {len(bundle.entry)}")
        print(f"   • Output directory: {output_dir}/")
        
        print(f"\n📋 Generated FHIR Resources:")
        print(f"   ✓ Bundle (collection of all resources)")
        print(f"   ✓ PlanDefinition (clinical protocol)")
        print(f"   ✓ ActivityDefinition ({len(activity_defs)} actions)")
        print(f"   ✓ Library (with Arden Syntax logic)")
        
        print(f"\n🎯 Use Cases:")
        print(f"   • Import into FHIR-compliant EHR systems")
        print(f"   • Clinical decision support integration")
        print(f"   • Guideline implementation and testing")
        print(f"   • Quality measurement and reporting")
        
        return {
            'extracted_data': extracted_data,
            'bundle': bundle,
            'plan_definition': plan_def,
            'activity_definitions': activity_defs,
            'library': library,
            'arden_syntax': arden_mlm
        }
if __name__ == "__main__":
    converter = MedicalDiagramToFHIR(llm_provider="openai")
    diagram_path = r"D:\HealthCare_ChatBot\fhir-rag-cpg\data\stroke_guideline.jpg"
    output_directory = r"D:\HealthCare_ChatBot\fhir-rag-cpg\output"
    
    converter.process_diagram(diagram_path, output_directory)