# Clinical Practice Guideline: Fever in the Returning Traveler

## Overview

**Disease/Condition:** Fever  
**Medical Domain:** Infectious Disease  
**Target Population:** Travelers returning with fever  
**Evidence Level:** B  
**Source:** CDC  

## Purpose

To identify potential causes of fever in returning travelers

## Directory Structure

```
.
├── bundle.json                  # Complete FHIR Bundle
├── plandefinition.json         # Main clinical protocol
├── questionnaire.json          # Clinical assessment form
├── library.json                # Logic library reference
├── logic.cql                   # CQL decision logic
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
- **PlanDefinition**: fever-in-the-returning-traveler
  - 5 clinical actions defined
  
- **Questionnaire**: questionnaire-fever-in-the-returning-traveler
  - 7 assessment items

- **Library**: library-fever-in-the-returning-traveler
  - Contains executable clinical logic

### Supporting Resources
- **ActivityDefinitions**: 0 actions
- **Measures**: 1 quality measures
- **Evidence**: 3 evidence items
- **EvidenceVariables**: 2 measurement variables
- **MeasureReports**: 1 report templates

## Clinical Workflow

### Entry Criteria

**Inclusion Criteria:**
- Recent travel abroad


### Decision Points

5 clinical decision points defined in the guideline.

### Clinical Actions

0 specific clinical actions defined.

## Quality Measures

### Fever Etiology Identification
- **Type**: outcome
- **Description**: Percentage of patients with fever who have an identified etiology.
- **Improvement**: increase


## Implementation Guide

### Using the PlanDefinition
1. Load the bundle.json into your FHIR server
2. Reference the PlanDefinition in your CDS hooks
3. Use the Questionnaire for data collection
4. Apply the decision logic from logic.cql

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

## Next Steps

1. **Validation**: Review extracted logic with clinical experts
2. **Testing**: Test decision logic with sample cases
3. **Integration**: Configure in your FHIR server/CDS system
4. **Monitoring**: Track measure reports and outcomes

## Generated

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Generator**: CPG Builder from Medical Diagram  
**FHIR Version**: 4.0.1  

## Support

For questions or issues with this CPG implementation, please review:
- FHIR PlanDefinition documentation
- CQL specification
- Your organization's clinical informatics team
