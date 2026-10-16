# MediCare Patient Follow-Up Agent

A LangGraph-based care-coordination decision-support agent that analyzes patient follow-up data from a CSV file.

The application uses:

- Python
- pandas
- Pydantic
- LangChain
- LangChain OpenAI
- LangGraph
- `MemorySaver` for in-memory thread state


---

## 1. Features

### Data loading and validation

The application validates:

- Required CSV columns.
- Unique patient IDs.
- Patient ID format.
- Numeric fields.
- Date fields.
- Non-negative age values.
- Non-negative days since last visit.
- Valid missed-appointment values.

### Exploratory analysis

The application displays:

- Dataset dimensions.
- Missing-value counts.
- Numeric summaries.
- Diagnosis distribution.
- Missed-appointment distribution.

### Model-facing data minimization

`patient_name` is excluded from all model-facing tool outputs.

The model receives only explicitly selected fields through:

- `get_patient`
- `get_missed_appointments`
- `search_patients`

The CSV may contain additional columns, but tools do not expose those columns unless they are deliberately added to the safe projection.

> Excluding `patient_name` does not make the remaining data anonymous. Patient IDs, diagnoses, medications, laboratory values, vital signs, and clinical notes are still sensitive health information.

### Tool governance

Only allow-listed tools can execute:

```python
TOOLS = [
    get_patient,
    get_missed_appointments,
    search_patients,
    get_cohort_summary,
]
```

Tool calls are checked against an allow-list before execution.

Unknown tools are blocked and returned as controlled error messages.

### Input validation

Pydantic validates:

- Patient ID format.
- Search argument types.
- Search limits.
- Required report fields.
- Allowed priority values.
- Minimum evidence length.
- Minimum follow-up recommendation length.
- Unexpected additional fields.

### Recommendation safety

The deterministic policy checker evaluates generated reports for:

- Medication start instructions.
- Medication stop instructions.
- Medication changes.
- Dose or dosage changes.
- Prescribing language.
- Potential diagnosis claims.
- Missing clinician-review language.
- Unknown patient IDs.

Invalid or unsafe reports are rejected and sent back to the model for retry.

### Traceability

An in-memory audit log records metadata for:

- Model calls.
- Tool calls.
- Tool errors.
- Blocked tool calls.
- Blocked reports.
- Approved reports.
- Fallback reports.
- Final parsing failures.

The audit log is designed to avoid storing complete clinical records, clinical notes, full prompts, or complete tool results.

### Human oversight

Approved reports must explicitly require qualified clinician review.

The agent is designed to recommend care-coordination actions only, such as:

- Contacting a patient.
- Coordinating appointment scheduling.
- Sending an appointment reminder.
- Arranging clinician review.
- Requesting repeat assessment through the clinical team.
- Escalating information to appropriate clinical staff.

### Operational resilience

`MAX_TURNS` prevents unbounded model, tool, and audit loops.

If the agent cannot produce a valid and safe report within the configured turn limit, it returns a safe fallback response with no patient-specific actions.

### Thread memory

`MemorySaver` stores state by `thread_id` during the active Python process or notebook session.

Use:

- A new `thread_id` for an independent analysis.
- The same `thread_id` for an intentional follow-up conversation.

Memory is lost when the Python process or notebook kernel stops.

---

## 2. Requirements

Recommended environment:

- Python 3.10 or later
- Pydantic 2.x
- pandas
- LangChain
- LangChain OpenAI
- LangGraph
- OpenAI API access

Install the dependencies:

```bash
pip install -U pandas pydantic langchain langchain-openai langgraph
```

---

## 3. Configure the OpenAI API key

The application requires the `OPENAI_API_KEY` environment variable.

### macOS/Linux

```bash
export OPENAI_API_KEY="your-api-key"
```

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

### Windows Command Prompt

```cmd
set OPENAI_API_KEY=your-api-key
```

You can verify that the variable exists without displaying the secret:

```python
import os

if os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY is configured")
else:
    print("OPENAI_API_KEY is missing")
```

Never place the API key directly in source code:

```python
# Do not do this:
OPENAI_API_KEY = "sk-..."
```

---

## 4. Project structure

A minimal script-based project may look like this:

```text
medicare-follow-up-agent/
├── patient_data.csv
├── medicare_agent.py
└── README.md
```

A notebook-based project may look like this:

```text
medicare-follow-up-agent/
├── patient_data.csv
├── medicare_follow_up_agent.ipynb
└── README.md
```

The `patient_data.csv` file must be in the same directory as the script or notebook.

---

## 5. CSV schema

The CSV must contain all of the following columns:

```text
patient_id
patient_name
age
gender
diagnosis
current_medication
lab_test
lab_value
lab_unit
last_visit_date
next_scheduled_visit
days_since_last_visit
missed_last_appointment
vitals_bp_systolic
vitals_bp_diastolic
vitals_heart_rate
vitals_spo2
notes
```

### Example CSV

The following data is fictional and provided for testing only:

```csv
patient_id,patient_name,age,gender,diagnosis,current_medication,lab_test,lab_value,lab_unit,last_visit_date,next_scheduled_visit,days_since_last_visit,missed_last_appointment,vitals_bp_systolic,vitals_bp_diastolic,vitals_heart_rate,vitals_spo2,notes
P0001,Example Patient,67,F,Hypertension,Medication A,HbA1c,7.2,%,2024-01-10,2024-04-10,90,No,146,88,78,97,Requires routine follow-up
P0002,Another Patient,54,M,Diabetes,Medication B,Glucose,185,mg/dL,2024-01-05,,120,Yes,138,84,82,96,Missed scheduled appointment
P0003,Sample Patient,72,F,Heart disease,Medication C,Creatinine,1.4,mg/dL,2024-02-01,2024-05-01,60,No,132,79,75,98,Clinician review may be appropriate
```

Do not use real patient data unless the environment has been approved for handling protected health information.

---

## 6. Data validation

### Required columns

If any required column is missing, application startup fails.

Example:

```text
ValueError: CSV is missing required columns: ['notes']
```

### Unique patient IDs

Every `patient_id` must be unique.

Example valid IDs:

```text
P0001
P0042
P12345
```

The expected format is:

```regex
^P\d{4,}$
```

Examples of invalid IDs:

```text
0001
patient-1
p-0001
P12
```

The application normalizes valid IDs to uppercase.

### Numeric fields

The following fields are converted to numeric values:

```text
age
lab_value
days_since_last_visit
vitals_bp_systolic
vitals_bp_diastolic
vitals_heart_rate
vitals_spo2
```

Invalid numeric values are converted to missing values and should be reviewed.

### Date fields

The following columns are converted using `pandas.to_datetime`:

```text
last_visit_date
next_scheduled_visit
```

Invalid dates become missing values.

### Missed appointment values

Accepted values are:

```text
Yes
No
blank
```

Values are compared case-insensitively after trimming whitespace.

Examples of invalid values:

```text
missed
true
1
unknown
```

---

## 7. Model-facing data minimization

The application uses an explicit safe-field projection.

`patient_name` is intentionally excluded from model-facing tool results.

Typical model-safe fields include:

```text
patient_id
age
gender
diagnosis
current_medication
lab_test
lab_value
lab_unit
last_visit_date
next_scheduled_visit
days_since_last_visit
missed_last_appointment
vitals_bp_systolic
vitals_bp_diastolic
vitals_heart_rate
vitals_spo2
notes
```

Do not expose all DataFrame columns directly to the model:

```python
# Avoid this for model-facing output:
result = df.copy()
```

Returning the entire DataFrame could expose:

- Patient names.
- Administrative fields.
- Future columns added without privacy review.
- Internal identifiers.
- Additional sensitive information.

### Clinical notes

Free-text notes may contain:

- Names.
- Telephone numbers.
- Addresses.
- Dates of birth.
- Other identifying information.
- Sensitive clinical details.

For production use, consider:

- Redacting identifiers from notes.
- Removing notes from general-purpose search.
- Returning notes only when specifically required.
- Applying a dedicated de-identification process.
- Reviewing model provider data-handling settings.

---

## 8. Available tools

### `get_patient`

Retrieves one model-safe patient record by patient ID.

Example request:

```text
Analyze patient P0003. Retrieve the full patient profile first.
```

The model may call:

```python
get_patient(patient_id="P0003")
```

If the patient does not exist, the tool returns a controlled error:

```json
{
  "error": "Patient ID P0003 was not found."
}
```

### `get_missed_appointments`

Retrieves patients whose last appointment was missed.

Example request:

```text
Find all patients who missed their last appointment and prioritize follow-up.
```

For missed-appointment requests, the system prompt instructs the model to call this tool first.

### `search_patients`

Searches patients using optional filters:

- Diagnosis.
- Laboratory test.
- Note keyword.
- Result limit.

Example request:

```text
Find patients with diabetes.
```

The search uses case-insensitive substring matching.

Search input is treated as literal text rather than a regular expression by using:

```python
regex=False
```

The search limit is restricted by Pydantic to a safe range.

### `get_cohort_summary`

Returns aggregate clinic context, including:

- Total patient count.
- Missed-appointment count.
- Diagnosis counts.
- Laboratory-test counts.
- Summary statistics for days since last visit.

Raw laboratory values from different tests or units must not be compared.

---

## 9. Structured output format

The agent must return valid JSON with exactly these top-level fields:

```json
{
  "summary": "Summary of the care-coordination review.",
  "prioritized_actions": [
    {
      "patient_id": "P0003",
      "priority": "Medium",
      "evidence": [
        "Fact returned by a tool."
      ],
      "recommended_follow_up": [
        "Care-coordination action requiring qualified clinician review."
      ]
    }
  ],
  "safety_note": "All recommendations require qualified clinician review."
}
```

### Priority values

Only the following values are allowed:

```text
High
Medium
Low
```

### Evidence requirements

Evidence must be based on facts returned by tools.

The model must not invent:

- Symptoms.
- Test results.
- Appointment dates.
- Medication adherence.
- Patient preferences.
- Clinical outcomes.
- Information not present in the tool output.

### Follow-up requirements

Recommended follow-up should be limited to care-coordination or clinician-review actions.

Examples:

```text
Contact the patient to coordinate appointment scheduling and request qualified clinician review.
```

```text
Send an appointment reminder and route the case to the care-coordination team for clinician review.
```

```text
Arrange a repeat assessment through the appropriate clinical team.
```

The agent must not recommend:

- Starting medication.
- Stopping medication.
- Changing medication.
- Changing medication dosage.
- Prescribing treatment.
- Making a new diagnosis.

---

## 10. Running the application

If the main file is named `medicare_agent.py`, run:

```bash
python medicare_agent.py
```

The application will:

1. Load `patient_data.csv`.
2. Validate the dataset.
3. Print exploratory statistics.
4. Run a single-patient analysis.
5. Run an optional MemorySaver follow-up.
6. Run a missed-appointment analysis.
7. Print the metadata-only audit log.

Example output:

```text
Dataset shape: (3, 18)

--- Task 4: Single Patient Analysis ---
{
  "summary": "...",
  "prioritized_actions": [
    {
      "patient_id": "P0003",
      "priority": "Medium",
      "evidence": [
        "..."
      ],
      "recommended_follow_up": [
        "..."
      ]
    }
  ],
  "safety_note": "All recommendations require qualified clinician review."
}
```

---

## 11. Running in a Jupyter notebook

For Jupyter:

1. Place `patient_data.csv` in the notebook directory.
2. Set `OPENAI_API_KEY` before starting the notebook or in the environment.
3. Run imports and configuration.
4. Run data loading and validation.
5. Run Pydantic schema definitions.
6. Run audit logging and tools.
7. Run the safety policy checker.
8. Compile the LangGraph workflow.
9. Run the agent examples.

For DataFrame display, you may use:

```python
from IPython.display import display

display(df.head())
```

The application does not require Jupyter and can also run as a normal Python script.

---

## 12. Using `run_agent`

The main helper function is:

```python
run_agent(
    request: str,
    thread_id: str | None = None,
)
```

### Independent analysis

Use a new thread ID for an independent request:

```python
import uuid

thread_id = f"review_{uuid.uuid4().hex}"

report = run_agent(
    request="Review patients who missed their last appointment.",
    thread_id=thread_id,
)

print(report)
```

### Single-patient analysis

```python
report = run_agent(
    request=(
        "Retrieve patient P0003. "
        "Summarize follow-up priority and provide a care-coordination plan "
        "requiring qualified clinician review."
    ),
    thread_id="patient_P0003_review",
)

print(report)
```

### Follow-up using the same memory thread

Reuse the same thread ID when the follow-up should use prior conversation state:

```python
follow_up = run_agent(
    request=(
        "Summarize only the evidence supporting the earlier priority. "
        "Do not add new conclusions."
    ),
    thread_id="patient_P0003_review",
)

print(follow_up)
```

### New conversation

Use a different thread ID for a separate request:

```python
new_report = run_agent(
    request="Review all patients who missed their last appointment.",
    thread_id="new_missed_appointment_review",
)
```

Do not reuse a thread ID across unrelated patients or unrelated workflows.

---

## 13. MemorySaver behavior

`MemorySaver` provides in-memory checkpointing by `thread_id`.

It provides:

- Conversation continuity.
- Thread-based state separation.
- State persistence during the active Python process.

It does not provide:

- Permanent storage.
- Database durability.
- Cross-process persistence.
- Automatic encryption.
- Backup and disaster recovery.
- Production-grade audit retention.

All memory is lost when the Python process or notebook kernel stops.

For production, use an approved persistent checkpointer with:

- Authentication.
- Authorization.
- Encryption.
- Retention policies.
- Backup and recovery.
- Monitoring.
- Patient or tenant isolation.

---

## 14. Audit logging

The application records metadata-only audit events.

View the audit log:

```python
events = get_audit_log()

for event in events:
    print(event)
```

Example event:

```json
{
  "timestamp": "2025-01-01T12:00:00+00:00",
  "event_type": "tool_call",
  "thread_id": "patient_P0003_review",
  "details": {
    "tool": "get_patient",
    "argument_keys": [
      "patient_id"
    ]
  }
}
```

Possible event types include:

```text
model_call
tool_call
tool_error
tool_blocked
blocked_report
approved_report
fallback_report
runner_rejection
runner_parse_failure
```

The audit log should not store:

- Patient names.
- Full clinical notes.
- Complete model prompts.
- Complete tool outputs.
- API keys.
- Passwords or credentials.
- Sensitive data that is not needed for traceability.

The current audit log is stored in memory and is lost when the process exits.

For production, use a secure, access-controlled, tamper-evident audit store.

---

## 15. Safety policy checks

The deterministic safety checker runs after Pydantic schema validation.

It checks for:

- Medication instructions.
- Prescribing instructions.
- Dose changes.
- Potential diagnosis claims.
- Missing clinician-review language.
- Unknown patient IDs.

Examples of blocked language:

```text
Increase the patient's medication dose immediately.
```

```text
Stop the patient's current medication.
```

```text
Prescribe a new medication.
```

```text
The patient has a confirmed new diagnosis.
```

Examples of acceptable care-coordination language:

```text
Contact the patient to coordinate a follow-up appointment and request qualified clinician review.
```

```text
Route the case to the appropriate clinical team for review.
```

```text
Arrange a repeat assessment through a qualified healthcare professional.
```

The policy checker is a prototype control. It is not a complete clinical safety system and is not a substitute for human review.

---

## 16. Fallback behavior

If the model cannot produce a valid and safe report within `MAX_TURNS`, the application returns:

```json
{
  "summary": "The agent could not produce a validated structured report within the configured turn limit.",
  "prioritized_actions": [],
  "safety_note": "No action should be taken from this output. Qualified clinician review is required."
}
```

The fallback response contains no patient-specific recommendations.

---

## 17. Error handling

### Missing CSV file

Error:

```text
FileNotFoundError: patient_data.csv must be in the same directory...
```

Fix:

```text
Place patient_data.csv beside the script or notebook.
```

### Missing API key

Error:

```text
EnvironmentError: Set OPENAI_API_KEY before running...
```

Fix:

```bash
export OPENAI_API_KEY="your-api-key"
```

### Missing CSV columns

Error:

```text
ValueError: CSV is missing required columns: [...]
```

Fix:

- Check column names.
- Check spelling.
- Check capitalization.
- Remove unexpected leading or trailing spaces.
- Confirm that all required columns exist.

### Duplicate patient IDs

Error:

```text
ValueError: patient_id must be unique.
```

Fix:

- Remove duplicate records.
- Investigate whether duplicates represent separate visits.
- If multiple visits are required, redesign the data model to use a separate visit identifier.

### Invalid patient ID

Error:

```text
ValueError: patient_id must match the format P0001.
```

Fix the CSV or request to use IDs such as:

```text
P0001
P0042
P12345
```

### Invalid missed-appointment value

Accepted values:

```text
Yes
No
blank
```

### Invalid model response

The report is rejected and the model is asked to retry using the required JSON structure and safety rules.

---

## 18. Security and privacy guidance

This application processes sensitive patient-related information.

Before using real patient data:

- Obtain organizational approval.
- Use an approved model deployment.
- Confirm privacy and data-processing requirements.
- Restrict access to authorized users.
- Protect API keys.
- Encrypt data in transit and at rest.
- Establish retention and deletion policies.
- Avoid logging patient data.
- Redact identifiers from free-text notes where possible.
- Separate development, testing, and production environments.
- Monitor model and tool access.
- Do not use real patient data in unapproved notebooks or personal environments.
- Do not commit `patient_data.csv` or API keys to Git.

Add the following to `.gitignore`:

```gitignore
.env
*.key
patient_data.csv
__pycache__/
.ipynb_checkpoints/
```

Never commit secrets:

```bash
git add .
git status
```

Before committing, verify that no patient data or credentials are included.

---

## 19. Clinical safety limitations

This system is intended only for care coordination and decision support.

It must not:

- Diagnose a patient.
- Prescribe medication.
- Recommend medication changes.
- Change medication dosage.
- Replace clinician assessment.
- Replace emergency services.
- Make autonomous treatment decisions.
- Be used as the sole basis for patient care.
- Directly modify medical records without authorized controls.
- Contact patients autonomously without an approved workflow.

Every generated report requires qualified clinician review.

If an emergency may be present, the appropriate emergency or clinical process must be followed. This agent is not an emergency response system.

---

## 20. Testing recommendations

Automated tests should cover:

- Missing CSV files.
- Missing required columns.
- Duplicate patient IDs.
- Invalid patient IDs.
- Invalid dates.
- Invalid numeric values.
- Invalid missed-appointment values.
- Patient-name exclusion.
- Tool allow-list enforcement.
- Tool argument validation.
- Unknown patient IDs.
- Unsafe medication recommendations.
- Diagnosis claims.
- Missing clinician-review language.
- Malformed JSON.
- Invalid structured output.
- Maximum-turn fallback.
- Thread isolation.
- Audit-event creation.
- Safe handling of tool errors.

Example test installation:

```bash
pip install pytest
```

Example tests:

```python
def test_patient_id_validation():
    valid = PatientLookup(patient_id="p0001")
    assert valid.patient_id == "P0001"


def test_invalid_patient_id():
    import pytest

    with pytest.raises(Exception):
        PatientLookup(patient_id="invalid-id")


def test_patient_name_is_not_model_facing():
    payload = json.loads(
        get_patient.invoke({"patient_id": "P0001"})
    )

    assert len(payload) == 1
    assert "patient_name" not in payload[0]


def test_priority_action_requires_evidence():
    import pytest

    with pytest.raises(Exception):
        PriorityAction(
            patient_id="P0001",
            priority="High",
            evidence=[],
            recommended_follow_up=[
                "Coordinate qualified clinician review."
            ],
        )


def test_policy_blocks_medication_change():
    report = AgentReport(
        summary="Follow-up is needed.",
        prioritized_actions=[
            PriorityAction(
                patient_id="P0001",
                priority="High",
                evidence=[
                    "The patient missed an appointment."
                ],
                recommended_follow_up=[
                    "Change the medication dose immediately."
                ],
            )
        ],
        safety_note="Qualified clinician review is required.",
    )

    violations = policy_violations(report)

    assert violations
```

---

## 21. Production recommendations

Before production use, consider adding the following.

### Authentication and authorization

- User authentication.
- Role-based access control.
- Patient-level authorization.
- Session expiration.
- Administrative review.
- Access logging.

### Persistent audit storage

- Secure database storage.
- Tamper-evident logs.
- Retention rules.
- Audit review workflows.
- Time synchronization.
- Restricted access.

### Model controls

- Approved model versions.
- Prompt versioning.
- Model change management.
- Input and output filtering.
- Rate limiting.
- Request timeouts.
- Retry limits.
- Cost monitoring.
- Provider data-retention review.

### Data controls

- De-identification.
- Field-level access restrictions.
- Notes redaction.
- Encryption.
- Data retention controls.
- Data-quality monitoring.
- Secure deletion.

### Clinical governance

- Clinician-authored policies.
- Formal review of prioritization criteria.
- False-positive testing.
- False-negative testing.
- Human approval workflows.
- Escalation procedures.
- Periodic safety review.
- Documentation of intended use and prohibited use.

### Operational controls

- Monitoring and alerting.
- Secure deployment pipeline.
- Dependency scanning.
- Secret management.
- Backup and recovery.
- Incident response.
- Versioned configuration.
- Separate development and production environments.

---

## 22. Recommended architecture

A safer deployment pattern is:

```text
Authenticated User
        |
        v
Care-Coordination Application
        |
        v
Authentication and Authorization
        |
        v
Allow-Listed LangGraph Agent
        |
        +--> Deterministic Patient-Data Tools
        |
        +--> Pydantic Output Validation
        |
        +--> Deterministic Safety Policy Checker
        |
        +--> Audit Logging
        |
        v
Qualified Clinician Review
        |
        v
Approved Care-Coordination Workflow
```

The model should not directly:

- Modify patient records.
- Change appointments without authorization.
- Send patient messages autonomously.
- Change medication.
- Create diagnoses.
- Trigger clinical treatment.
- Make emergency decisions.

---

## 23. Limitations of the current prototype

The current implementation has the following limitations:

- `MemorySaver` is not persistent.
- The audit log is stored only in memory.
- Policy checks are pattern-based and may produce false positives or false negatives.
- Free-text clinical notes may contain sensitive information.
- The model may still make factual or reasoning errors.
- The output requires human review.
- The system does not independently verify the accuracy of the source CSV.
- The system does not replace clinical triage.
- The system has no built-in authentication or authorization.
- The system does not guarantee regulatory compliance.
- The system does not guarantee HIPAA, GDPR, or other legal compliance.
- The system is not a certified medical device.

---
