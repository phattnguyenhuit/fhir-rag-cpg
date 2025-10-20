library DiabetesManagement version '1.0.0' 
using FHIR version '4.0.1'
include FHIRHelpers version '1.0.0'

context Patient

// Value Sets
valueset "Diabetes Symptoms" : 'http://example.org/fhir/ValueSet/diabetes-symptoms'
valueset "Glucose Test Results" : 'http://example.org/fhir/ValueSet/glucose-test-results'

// Codes
code "Diabetes Mellitus Type 2" : '250.00' from "ICD-10-CM"
code "Fasting Glucose" : '155' from "LOINC"

// Reusable Definitions
define "HasClassicSymptoms":
    exists (Observation O
        where O.code in "Diabetes Symptoms"
        and O.subject = Patient
        and O.status = 'final'
    )

define "FastingGlucoseTest":
    exists (Observation O
        where O.code = "Fasting Glucose"
        and O.subject = Patient
        and O.status = 'final'
        and O.valueQuantity.value >= 126 // Fasting glucose >= 126 mg/dL
    )

define "Recommendations":
    if "HasClassicSymptoms" then
        'Consider further testing for diabetes mellitus type 2.'
    else if "FastingGlucoseTest" then
        'Diagnosis of diabetes mellitus type 2 confirmed based on fasting glucose levels.'
    else
        'No specific recommendations at this time.'
    endif

// Main Logic
define "ClinicalDecisionSupport":
    let recommendations = "Recommendations"
    return recommendations

// Alert Logic
define "AlertNeeded":
    count("ClinicalDecisionSupport") > 0

define "AlertMessage":
    if "AlertNeeded" then
        'Clinical Decision Support: ' || count("ClinicalDecisionSupport") || ' recommendations'
    else
        'No specific recommendations at this time'
    endif

// Output
define "Output":
    if "AlertNeeded" then
        write "Alert Message: " || "AlertMessage"
        write "Recommendations: " || "ClinicalDecisionSupport"
    endif