# Clinical Practice Guideline: Acute Ischemic Stroke Management Algorithm

## Overview

**Disease/Condition:** Acute Ischemic Stroke  
**Medical Domain:** Neurology  
**Target Population:** Patients with acute ischemic stroke symptoms  
**Evidence Level:** A  
**Source:** American Heart Association  

## Purpose

Determine treatment options for acute ischemic stroke

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
- **PlanDefinition**: acute-ischemic-stroke-management-algorithm
  - 5 clinical actions defined
  
- **Questionnaire**: questionnaire-acute-ischemic-stroke-management-algorithm
  - 9 assessment items

- **Library**: library-acute-ischemic-stroke-management-algorithm
  - Contains executable clinical logic

### Supporting Resources
- **ActivityDefinitions**: 1 actions
- **Measures**: 1 quality measures
- **Evidence**: 5 evidence items
- **EvidenceVariables**: 5 measurement variables
- **MeasureReports**: 1 report templates

## Clinical Workflow

### Entry Criteria

**Inclusion Criteria:**
- Symptoms of stroke
- Head CT shows ischemic changes

**Exclusion Criteria:**
- Head CT shows hemorrhage
- Symptoms not consistent with stroke


### Decision Points

5 clinical decision points defined in the guideline.

### Clinical Actions

1 specific clinical actions defined.

## Quality Measures

### Time to Treatment
- **Type**: process
- **Description**: Time from symptom onset to treatment initiation
- **Improvement**: decrease


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
