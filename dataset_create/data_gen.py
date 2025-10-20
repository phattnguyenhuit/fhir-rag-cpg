from fhir.resources.plandefinition import PlanDefinition
from fhir.resources.triggerdefinition import TriggerDefinition
from fhir.resources.timing import Timing
from datetime import datetime
import re
import spacy

class ClinicalGuidelineConverter:
    """Convert natural language clinical guidelines to FHIR PlanDefinition and Arden Syntax"""
    
    def __init__(self):
        # Load spaCy for basic NLP (install with: pip install spacy)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_clinical_elements(self, text):
        """Extract key clinical elements using regex and simple NLP"""
        elements = {
            'condition': None,
            'action': None,
            'timing': None,
            'frequency': None
        }
        
        # Extract condition (e.g., "patients with diabetes")
        condition_match = re.search(r'patients? with ([^,\.]+)', text, re.IGNORECASE)
        if condition_match:
            elements['condition'] = condition_match.group(1).strip()
        
        # Extract action verbs
        action_match = re.search(r'should (have|be|receive) ([^,\.]+)', text, re.IGNORECASE)
        if action_match:
            elements['action'] = action_match.group(2).strip()
        
        # Extract timing/frequency
        timing_patterns = [
            (r'every (\d+) (day|week|month|year)s?', 'periodic'),
            (r'(\d+) times? per (day|week|month|year)', 'frequency'),
            (r'(daily|weekly|monthly|annually)', 'periodic')
        ]
        
        for pattern, timing_type in timing_patterns:
            timing_match = re.search(pattern, text, re.IGNORECASE)
            if timing_match:
                elements['timing'] = timing_match.group(0)
                elements['frequency'] = timing_type
                break
        
        return elements
    
    def text_to_plandefinition(self, text, title="Clinical Guideline"):
        """Convert clinical text to FHIR PlanDefinition"""
        elements = self.extract_clinical_elements(text)
        
        # Build action with proper structure
        action_dict = {
            "title": elements.get('action', 'Clinical Action'),
            "description": text,
            "textEquivalent": text
        }
        
        # Add condition if present
        if elements['condition']:
            action_dict['condition'] = [{
                "kind": "applicability",
                "expression": {
                    "language": "text/cql",
                    "expression": f"Patient has {elements['condition']}"
                }
            }]
        
        # Add timing if present
        if elements['timing']:
            action_dict['timingDuration'] = {
                "value": elements['timing']
            }
        
        # Create PlanDefinition
        plan_def = PlanDefinition.construct(
            id="cpg-" + re.sub(r'\W+', '-', title.lower()),
            title=title,
            status="draft",
            type={
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/plan-definition-type",
                    "code": "clinical-protocol"
                }]
            },
            date=datetime.now().isoformat(),
            description=f"Generated from: {text}",
            action=[action_dict]
        )
        
        return plan_def, elements
    
    def plandefinition_to_arden(self, plan_def, elements):
        """Convert FHIR PlanDefinition to Arden Syntax MLM"""
        
        # Extract key information
        title = plan_def.title.replace(" ", "_")
        action = plan_def.action[0] if plan_def.action else {}
        description = action.get('description', '')
        
        # Build condition logic
        condition_logic = "true"
        if elements.get('condition'):
            condition_logic = f"diagnosis includes '{elements['condition']}'"
        
        # Build timing logic
        timing_logic = ""
        if elements.get('timing'):
            timing_logic = f"\n    /* Check interval: {elements['timing']} */"
        
        # Generate Arden Syntax MLM
        arden_mlm = f"""maintenance:
    title: {title};;
    mlmname: {title.lower()};;
    arden: Version 2.10;;
    version: 1.0;;
    institution: Generated;;
    author: Automated Conversion;;
    specialist: ;;
    date: {datetime.now().strftime('%Y-%m-%d')};;
    validation: testing;;

library:
    purpose: {plan_def.description or 'Clinical decision support'};;
    explanation: Generated from FHIR PlanDefinition;;
    keywords: clinical guideline; decision support;;
    citations: ;;

knowledge:
    type: data-driven;;
    data:
        /* Patient data elements */
        diagnosis := read {{diagnosis}};
        last_test_date := read {{last_test_date for '{action.get('title', 'test')}'}};
        {timing_logic}
    ;;
    
    evoke:
        /* Trigger when patient record is accessed */
        patient_encounter
    ;;
    
    logic:
        if {condition_logic} then
            recommend_action := true;
            message := "{action.get('title', 'Perform clinical action')}";
        else
            recommend_action := false;
        endif;
    ;;
    
    action:
        if recommend_action then
            write message;
            return message;
        endif;
    ;;

end:
"""
        return arden_mlm
    
    def convert_guideline(self, text, title="Clinical Guideline", output_file=None):
        """Complete conversion pipeline"""
        
        # Step 1: Convert to FHIR
        plan_def, elements = self.text_to_plandefinition(text, title)
        
        # Step 2: Convert to Arden Syntax
        arden_mlm = self.plandefinition_to_arden(plan_def, elements)
        
        # Step 3: Save outputs
        if output_file:
            # Save MLM
            with open(f"{output_file}.mlm", "w") as f:
                f.write(arden_mlm)
            
            # Save FHIR JSON
            with open(f"{output_file}.json", "w") as f:
                f.write(plan_def.json(indent=2))
            
            print(f"✓ Generated {output_file}.mlm")
            print(f"✓ Generated {output_file}.json")
        
        return {
            'plan_definition': plan_def,
            'arden_syntax': arden_mlm,
            'extracted_elements': elements
        }


# Example usage
if __name__ == "__main__":
    converter = ClinicalGuidelineConverter()
    
    # Test case 1: Diabetes monitoring
    text1 = "Patients with diabetes should have HbA1c checked every 3 months."
    result1 = converter.convert_guideline(
        text1, 
        title="Diabetes HbA1c Monitoring",
        output_file="diabetes_hba1c"
    )
    
    print("\n" + "="*60)
    print("EXTRACTED ELEMENTS:")
    print("="*60)
    for key, value in result1['extracted_elements'].items():
        print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("ARDEN SYNTAX MLM (first 500 chars):")
    print("="*60)
    print(result1['arden_syntax'][:500] + "...")
    
    # Test case 2: Hypertension screening
    text2 = "Adults should be screened for hypertension annually."
    result2 = converter.convert_guideline(
        text2,
        title="Hypertension Screening",
        output_file="hypertension_screening"
    )