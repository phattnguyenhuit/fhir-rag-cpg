library acute_ischemic_stroke_management_algorithm version '1.0.0'

using FHIR version '4.0.1'

include FHIRHelpers version '4.0.1'

// Clinical Practice Guideline: Acute Ischemic Stroke Management Algorithm
// Domain: Neurology
// Disease: Acute Ischemic Stroke

// Contexts
context Patient

// =============================================================
// Value Sets and Codes
// =============================================================

// Codes for: Symptom onset
// SNOMED: 431855005
// ICD10: I63.9
// LOINC: LP15000-8

// Codes for: IV tPA Administration

// Codes for: Symptom onset

// Codes for: IV tPA Administration

// Codes for: Endovascular Therapy Decision

// =============================================================
// Data Elements
// =============================================================
define "Patient Age":
  AgeInYears()

define "Patient Gender":
  Patient.gender

// =============================================================
// Inclusion Criteria
// =============================================================
define "Meets Inclusion Criteria":
  // Symptoms of stroke
  // Head CT shows ischemic changes
  true // TODO: Implement actual logic

// =============================================================
// Decision Logic
// =============================================================

define "Node1":
  // What is the time since symptom onset?
  // Condition: Symptom onset <= 3 hours
  false // TODO: Implement actual logic

define "Node2":
  // Is IV tPA criteria met?
  // Condition: Yes
  false // TODO: Implement actual logic

define "Node3":
  // Is symptom onset between 3 to 4.5 hours?
  // Condition: 3 < symptom onset <= 4.5
  false // TODO: Implement actual logic

define "Node4":
  // Is IV tPA administered?
  // Condition: Yes
  false // TODO: Implement actual logic

define "Node5":
  // Is endovascular therapy indicated?
  // Condition: Discuss with vascular neurologist
  false // TODO: Implement actual logic

// =============================================================
// Recommendations
// =============================================================
define "Active Recommendations":
  // Compile all applicable recommendations
  {} // TODO: Implement recommendation logic
