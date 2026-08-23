import { useCallback, useRef, useState } from "react";
import { sendChat } from "@/lib/api";
import type { ChatMessage as ChatMessageType } from "@/types";
import { ChatMessage, TypingIndicator } from "@/components/chat/ChatMessage";
import { Composer, WelcomeState } from "@/components/chat/Composer";

function newSessionId(): string {
  return crypto.randomUUID();
}

export default function App() {
  const sessionIdRef = useRef<string>(newSessionId());
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [pending, setPending] = useState(false);
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, []);

  const send = useCallback(
    async (text: string) => {
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setPending(true);
      scrollToBottom();
      try {
        const response = await sendChat(text, sessionIdRef.current);
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            content: response.final_answer,
            sources: response.sources_cited,
            handoff: response.handoff_recommended,
          },
        ]);
      } catch (error) {
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            content:
              error instanceof Error
                ? error.message
                : "Something went wrong. Please try again.",
            error: true,
          },
        ]);
      } finally {
        setPending(false);
        scrollToBottom();
      }
    },
    [scrollToBottom],
  );

  const startNewConversation = useCallback(() => {
    sessionIdRef.current = newSessionId();
    setMessages([]);
  }, []);

  return (
    <div className="mx-auto flex h-dvh max-w-3xl flex-col">
      <header className="flex items-center justify-between border-b border-border px-5 py-5">
        <div className="flex items-baseline gap-3">
          <span className="font-display text-2xl font-medium tracking-tight">
            Aster &amp; Row
          </span>
          <span className="hidden font-mono text-[0.72rem] uppercase tracking-[0.18em] text-muted-foreground sm:inline">
            Support Concierge
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={startNewConversation}
            className="rounded-lg border border-border bg-card px-3.5 py-2 text-[0.875rem] font-medium shadow-sm transition-colors hover:border-primary/40 hover:bg-secondary/60 focus-visible:outline-2 focus-visible:outline-ring"
          >
            New conversation
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        {messages.length === 0 && !pending ? (
          <WelcomeState onExample={send} />
        ) : (
          <div className="flex flex-col gap-4 px-4 py-6 sm:px-6">
            {messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))}
            {pending && <TypingIndicator />}
            <div ref={scrollAnchorRef} />
          </div>
        )}
      </main>

      <Composer onSend={send} disabled={pending} />
    </div>
  );
}
