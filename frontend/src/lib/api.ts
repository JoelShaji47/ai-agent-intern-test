import type { AgentResponse } from "@/types";

const API_BASE =
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function sendChat(
  message: string,
  sessionId: string,
): Promise<AgentResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
  } catch {
    throw new Error(
      "Cannot reach the support agent. Check that the backend server is running.",
    );
  }
  if (!response.ok) {
    let detail = "The support agent is temporarily unavailable.";
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* keep default detail */
    }
    throw new Error(detail);
  }
  return (await response.json()) as AgentResponse;
}
