export interface ToolCall {
  tool: string;
  arguments: Record<string, unknown>;
}

export interface AgentResponse {
  final_answer: string;
  sources_cited: string[];
  tool_calls_made: ToolCall[];
  handoff_recommended: boolean;
}

export interface ChatMessage {
  role: "user" | "agent";
  content: string;
  sources?: string[];
  handoff?: boolean;
  error?: boolean;
}
