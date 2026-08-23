import { useEffect, useRef, type FormEvent, type KeyboardEvent } from "react";

const EXAMPLES = [
  "How long is your return window?",
  "Where is my order?",
  "Can I put the Breeze Tumbler in the dishwasher?",
];

export function WelcomeState({ onExample }: { onExample: (text: string) => void }) {
  return (
    <div className="flex flex-col items-center px-4 py-14 text-center">
      <p className="font-mono text-[0.62rem] uppercase tracking-[0.22em] text-muted-foreground">
        Aster &amp; Row
      </p>
      <h1 className="font-display mt-3 max-w-md text-balance text-3xl font-medium leading-snug">
        Support for the long way round.
      </h1>
      <p className="mt-3 max-w-sm text-[0.9rem] leading-relaxed text-muted-foreground">
        Ask about returns, shipping, warranty, or your order status — answers
        come straight from our policy library.
      </p>
      <div className="mt-8 grid w-full max-w-lg gap-2 sm:grid-cols-1">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            onClick={() => onExample(example)}
            className="group rounded-xl border border-border bg-card px-4 py-3 text-left text-[0.88rem] shadow-sm transition-colors hover:border-primary/40 hover:bg-secondary/60 focus-visible:outline-2 focus-visible:outline-ring"
          >
            {example}
            <span
              aria-hidden
              className="float-right text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-primary"
            >
              →
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

const MAX_TEXTAREA_HEIGHT = 160;

export function Composer({
  onSend,
  disabled,
}: {
  onSend: (text: string) => void;
  disabled: boolean;
}) {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!disabled) inputRef.current?.focus();
  }, [disabled]);

  function autoResize() {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
  }

  function handleSubmit(event?: FormEvent) {
    event?.preventDefault();
    const value = inputRef.current?.value.trim();
    if (!value || disabled) return;
    if (inputRef.current) {
      inputRef.current.value = "";
      autoResize();
    }
    onSend(value);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t border-border bg-background/90 px-4 pt-4 pb-6 backdrop-blur"
    >
      <div className="mx-auto flex max-w-2xl items-end gap-2">
        <textarea
          ref={inputRef}
          rows={1}
          placeholder="Ask about returns, orders, warranty…"
          disabled={disabled}
          onChange={autoResize}
          onKeyDown={handleKeyDown}
          className="min-h-[52px] flex-1 resize-none rounded-xl border border-input bg-card px-5 py-3.5 text-[0.95rem] leading-relaxed shadow-sm outline-none transition-colors placeholder:text-muted-foreground focus:border-primary/50 focus:ring-2 focus:ring-ring/30 disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={disabled}
          aria-label="Send message"
          className="inline-flex h-[52px] items-center gap-1.5 rounded-xl bg-primary px-5 text-[0.9rem] font-medium text-primary-foreground shadow-sm transition-all hover:brightness-110 focus-visible:outline-2 focus-visible:outline-ring disabled:cursor-not-allowed disabled:opacity-55"
        >
          Send
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="size-4" aria-hidden>
            <path d="m5 12 14 0" />
            <path d="m12 5 7 7-7 7" />
          </svg>
        </button>
      </div>
    </form>
  );
}
