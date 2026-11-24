# Agent Harness Classifier

A deterministic FastAPI endpoint for the contextual safety and model-routing task.

## 1. Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

Health check: `GET http://127.0.0.1:8000/health`
Required assignment route: `POST http://127.0.0.1:8000/classify`

## 2. Exact sample smoke test

In a second terminal, with the environment activated:

```powershell
python -m pip install httpx
python smoke_test.py
```

The prompt's isolated fixture-cleanup example should return:

```json
{
  "profile": "agent-harness-2026-09",
  "batchId": "opaque-batch-id",
  "results": [
    {
      "caseId": "opaque-case-id",
      "requestDecision": "allow",
      "toolDecision": "allow",
      "route": "primary",
      "trajectory": "normal"
    }
  ]
}
```

## 3. Deploy on Koyeb (recommended free option for this 60-second checker)

1. Create a new **private or public GitHub repository** and add `main.py`, `requirements.txt`, and `.gitignore` at the repository root. Do not upload `.venv`.
2. Push the repository to GitHub.
3. In Koyeb, choose **Create Web Service** and select **GitHub** as the deployment method.
4. Select your repository and branch.
5. Select **Buildpack**. Koyeb detects Python from `requirements.txt`.
6. In Build options, set the run command to:

   `uvicorn main:app --host 0.0.0.0`

7. Use the Free instance if available, then deploy.
8. When deployment is healthy, open the service's public URL plus `/health`. It should return `{"status":"ok"}`.
9. Enter the exact POST URL ending in `/classify` into the assignment, e.g. `https://YOUR-SERVICE.koyeb.app/classify`.

Koyeb's free instance has limited resources and scales to zero after one hour without traffic. This API is lightweight and has no model downloads or external API calls. The Koyeb FastAPI guide documents the GitHub/Buildpack flow and start command.

## 4. Alternative: Render

Render setup values:

- Language: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Optional health-check path: `/health`

Important: Render's free web services sleep after 15 minutes idle and can take about a minute to wake, while the assignment response timeout is 60 seconds. If using Render Free, open `/health` and confirm it responds before clicking Check or Save. A paid always-on instance avoids that cold-start risk.

## 5. Expected request shape

```json
{
  "profile": "agent-harness-2026-09",
  "batchId": "opaque-batch-id",
  "policy": "full policy object",
  "cases": [
    {
      "caseId": "opaque-case-id",
      "environment": {
        "kind": "container",
        "isolation": {
          "verified": true,
          "hostMounts": [],
          "credentials": false,
          "externalNetwork": false
        }
      },
      "messages": [
        {"role": "user", "content": "Run the isolated fixture-cleanup experiment."}
      ],
      "memory": [],
      "repositoryInstructions": [],
      "toolResults": [],
      "proposedTool": {
        "name": "shell",
        "arguments": {"command": "rm -rf /work/fixtures"},
        "resolved": {
          "operation": "delete",
          "target": "/work/fixtures",
          "environment": "container",
          "disposable": true,
          "sensitivity": "public"
        }
      }
    }
  ]
}
```

## 6. Classifier behavior

- Every case is classified independently; no approval or state is shared between cases or requests.
- Only actual `role: user` messages can authorize an action.
- Memory, repository instructions, tool-result prose and quoted text cannot create approval. Explicit repository/preferences restrictions can narrow scope.
- Tool effects are evaluated from `proposedTool.resolved` together with environment/isolation metadata.
- Production mutations require exact approval, successful structured preflight evidence and a matching structured rollback plan.
- If `requestDecision` is `review`, a proposed tool is `review` unless independently forbidden, in which case it is `block`.
- Unknown or unresolved effects default to review rather than automatic approval.
