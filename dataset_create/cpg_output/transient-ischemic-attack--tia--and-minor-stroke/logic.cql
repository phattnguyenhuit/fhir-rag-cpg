library Guideline_for_Diagnosis_and_Treatment_of_Transient_Ischemic_Attack_and_Minor_Stroke version '1.0.0'

using FHIR version '4.0.1'
include FHIRHelpers version '1.0.0'

context Patient

// Value Sets
valueset "Focal Neurological Signs" : 'http://example.org/fhir/ValueSet/focal-neurological-signs'
valueset "Imaging Findings" : 'http://example.org/fhir/ValueSet/imaging-findings'

// Definitions
define "FocalNeurologicalSignsPresent":
    exists (Observation O where O.code in "Focal Neurological Signs" and O.status = 'final' and O.subject = Patient)

define "ImagingShowsIschemicChanges":
    exists (Observation O where O.code in "Imaging Findings" and O.status = 'final' and O.subject = Patient)

// Decision Logic
define "Recommendations":
    if "FocalNeurologicalSignsPresent" then
        if "ImagingShowsIschemicChanges" then
            return ["Consider hospitalization for further evaluation and management of TIA."]
        else
            return ["Consider outpatient follow-up and risk factor modification."]
        endif
    else
        return []
    endif

// Alert Logic
define "AlertNeeded":
    count("Recommendations") > 0

define "AlertMessage":
    if "AlertNeeded" then
        "Clinical Decision Support: " + count("Recommendations") + " recommendations"
    else
        "No specific recommendations at this time"
    endif

// Main Output
define "DecisionPath":
    if "AlertNeeded" then
        "Recommendations available based on clinical findings."
    else
        "No recommendations available."
    endif

// Output
define "Output":
    if "AlertNeeded" then
        {
            "alert_message": "AlertMessage",
            "decision_path": "DecisionPath",
            "recommendations": "Recommendations"
        }
    else
        {
            "alert_message": "AlertMessage",
            "decision_path": "DecisionPath",
            "recommendations": []
        }
    endif