import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage as ChatMessageType } from "@/types";

/** Matches a knowledge-base citation token wherever it appears, consuming any
 * surrounding wrapper (square brackets or backticks) so bracketed,
 * backtick-wrapped, bare, and comma-combined forms all normalize the same. */
const CITE_RE = /(?:\[\s*|`)?([a-z0-9][a-z0-9.-]*\.md#[a-z0-9-]+)(?:\s*\]|`)?/g;

/** The 14 real knowledge-base documents. Citations naming anything else are
 * hallucinated/malformed (e.g. a transposed prefix like 05-final-sale...) and
 * are rendered as plain text instead of a chip, both inline and in pills. */
const KNOWN_DOCS = new Set([
  "01-returns-policy-current.md",
  "02-returns-policy-legacy.md",
  "03-final-sale-and-promotions.md",
  "04-damaged-or-wrong-items.md",
  "05-domestic-shipping.md",
  "06-international-shipping.md",
  "07-warranty.md",
  "08-order-changes-and-cancellations.md",
  "09-trailplus-membership.md",
  "10-gift-cards-and-price-adjustments.md",
  "11-product-care.md",
  "12-breeze-tumbler-product-card.md",
  "13-support-escalation.md",
  "14-internal-content-migration-notes.md",
]);

/** True when a citation token's filename portion (before #anchor) names one
 * of the 14 real knowledge-base documents. */
function isKnownCitation(cite: string): boolean {
  return KNOWN_DOCS.has(cite.split("#")[0]);
}

function withCiteChips(text: string): string {
  return text.replace(CITE_RE, (match, fname: string) =>
    isKnownCitation(fname) ? `\`${fname}\`` : match,
  );
}

/** Filenames actually cited inline in the answer text, deduplicated in
 * first-appearance order. Deliberately NOT the API's sources_cited field,
 * which records everything retrieved by search (superset). */
function citedSources(text: string): string[] {
  const seen = new Set<string>();
  const sources: string[] = [];
  for (const match of text.matchAll(CITE_RE)) {
    const cite = match[1];
    if (!isKnownCitation(cite) || seen.has(cite)) continue;
    seen.add(cite);
    sources.push(cite);
  }
  return sources;
}

function SourcePills({ sources }: { sources: string[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-3 border-t border-border/70 pt-2.5">
      <p className="font-mono text-[0.62rem] uppercase tracking-[0.14em] text-muted-foreground mb-1.5">
        Sources
      </p>
      <div className="flex flex-wrap gap-1.5">
        {sources.map((source) => (
          <span
            key={source}
            className="inline-flex items-center gap-1 rounded-md bg-secondary px-2 py-0.5 font-mono text-[0.66rem] text-secondary-foreground"
            title={source}
          >
            <span className="size-1 rounded-full bg-primary/60" aria-hidden />
            {source}
          </span>
        ))}
      </div>
    </div>
  );
}

function HandoffTicket() {
  return (
    <div
      role="status"
      className="handoff-ticket mt-3 flex items-start gap-2.5 rounded-lg px-3.5 py-2.5"
    >
      <svg
        className="mt-0.5 size-4 shrink-0 text-(--brass-deep)"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <path d="M13 16h-1v-4h-1" />
        <circle cx="12" cy="12" r="9" />
        <path d="M12 8h.01" />
      </svg>
      <div>
        <p className="text-[0.8rem] font-semibold text-(--brass-deep)">
          Human handoff recommended
        </p>
        <p className="text-[0.78rem] text-(--brass-deep)/80">
          This conversation was flagged for review by a support specialist.
        </p>
      </div>
    </div>
  );
}

export function TypingIndicator() {
  return (
    <div className="msg-in flex" aria-label="Agent is typing">
      <div className="rounded-2xl rounded-tl-sm border border-border bg-card px-4 py-3 shadow-sm">
        <div className="flex gap-1">
          <span className="typing-dot size-1.5 rounded-full bg-primary/70" />
          <span className="typing-dot size-1.5 rounded-full bg-primary/70" />
          <span className="typing-dot size-1.5 rounded-full bg-primary/70" />
        </div>
      </div>
    </div>
  );
}

export function ChatMessage({ message }: { message: ChatMessageType }) {
  const isUser = message.role === "user";
  if (isUser) {
    return (
      <div className="msg-in flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-[0.92rem] leading-relaxed text-primary-foreground shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.error) {
    return (
      <div className="msg-in flex">
        <div className="max-w-[90%] rounded-2xl rounded-tl-sm border border-destructive/30 bg-destructive/5 px-4 py-3 text-[0.88rem] leading-relaxed text-destructive">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="msg-in flex flex-col items-start">
      <div className="max-w-[92%] rounded-2xl rounded-tl-sm border border-border bg-card px-4 py-3 shadow-sm">
        <div className="chat-body text-[0.92rem] leading-relaxed [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_p+p]:mt-2 [&_strong]:font-semibold [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {withCiteChips(message.content)}
          </ReactMarkdown>
        </div>
        <SourcePills sources={citedSources(message.content)} />
        {message.handoff && <HandoffTicket />}
      </div>
    </div>
  );
}
