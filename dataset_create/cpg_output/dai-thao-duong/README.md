# Clinical Practice Guideline: Hướng dẫn chẩn đoán và điều trị đái tháo đường típ 2

## Overview

**Disease/Condition:** Đái tháo đường típ 2  
**Medical Domain:** Nội tiết  
**Target Population:** Người lớn và trẻ em có nguy cơ mắc đái tháo đường típ 2  
**Evidence Level:** A  
**Source:** Bộ Y tế Việt Nam  
**Language:** Vietnamese

## Purpose

Chẩn đoán và điều trị đái tháo đường típ 2

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
- **PlanDefinition**: h--ng-d-n-ch-n--o-n-v---i-u-tr----i-th-o----ng-t-p-2
  - 5 clinical actions defined
  
- **Questionnaire**: questionnaire-h--ng-d-n-ch-n--o-n-v---i-u-tr----i-th-o----ng-t-p-2
  - 8 assessment items

- **Library**: library-h--ng-d-n-ch-n--o-n-v---i-u-tr----i-th-o----ng-t-p-2
  - Contains executable clinical logic in CQL and Arden Syntax

### Supporting Resources
- **ActivityDefinitions**: 1 actions
- **Measures**: 1 quality measures
- **Evidence**: 5 evidence items
- **EvidenceVariables**: 2 measurement variables
- **MeasureReports**: 1 report templates

## Clinical Workflow

### Entry Criteria

**Inclusion Criteria:**
- Người lớn có BMI ≥ 23 kg/m2
- Có yếu tố nguy cơ như tiền sử gia đình, tăng huyết áp, rối loạn lipid máu

**Exclusion Criteria:**
- Không có triệu chứng đái tháo đường
- Không có yếu tố nguy cơ


### Decision Points

5 clinical decision points defined in the guideline.

### Clinical Actions

1 specific clinical actions defined.

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

### Tỷ lệ bệnh nhân được chẩn đoán đái tháo đường
- **Type**: process
- **Description**: Tỷ lệ bệnh nhân được chẩn đoán đúng theo tiêu chuẩn
- **Improvement**: increase


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
