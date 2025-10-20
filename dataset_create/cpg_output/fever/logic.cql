library fever_in_the_returning_traveler version '1.0.0'

using FHIR version '4.0.1'

include FHIRHelpers version '4.0.1'

// Clinical Practice Guideline: Fever in the Returning Traveler
// Domain: Infectious Disease
// Disease: Fever

// Contexts
context Patient

// Value Sets and Codes

// Codes for: Incubation < 21 days

// Codes for: Consider causes

// Codes for: Incubation > 21 days

// Codes for: decision

// Codes for: decision


// Data Elements
define "Patient Age":
  AgeInYears()

define "Patient Gender":
  Patient.gender

// Inclusion Criteria
define "Meets Inclusion Criteria":
  // Recent travel abroad
  true // Implement actual logic

// Decision Logic

define "Node1":
  // Is the incubation period less than 21 days?
  // Condition: Incubation < 21 days
  false // Implement actual logic

define "Node2":
  false // Implement actual logic

define "Node3":
  // Is the incubation period greater than 21 days?
  // Condition: Incubation > 21 days
  false // Implement actual logic

define "Action1":
  false // Implement actual logic

define "Action2":
  false // Implement actual logic


// Recommendations
define "Active Recommendations":
  // Compile all applicable recommendations
  {} // Implement recommendation logic
