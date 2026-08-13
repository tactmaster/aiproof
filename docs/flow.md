# aiproof flow

How a trigger becomes corrected text. Every branch below is pinned by a test
in `tests/test_flow_*.py` (run `make test-flow` for a route-by-route log
report in `build/flow-report.md`).

Both charts are flowcharts because the branch structure is the point; a
sequence diagram would show the clipboard/keystroke timing better but needs
one diagram per route.

## The simple version

```mermaid
flowchart TD
    IN["Hotkey / tray / CLI"] --> D["Daemon"]
    D --> M{"Mode?"}
    M -->|"auto (ydotool)"| CP["Copy selection<br/>(synthetic Ctrl+C)"]
    M -->|"clipboard-only<br/>or degraded"| PS["Read primary selection"]
    CP --> LLM["LLM proofread<br/>+ provider fallback"]
    PS --> LLM
    LLM --> G{"Safety guards<br/>pass?"}
    G -->|"changed"| OUT["Paste in place /<br/>copy to clipboard"]
    G -->|"unchanged"| CLEAN["Notify: clean, or<br/>'couldn't apply safely'"]
    OUT --> DONE["Notify: corrected<br/>+ restore old clipboard"]
```

## The detailed version

`🔔` = desktop notification (the exact strings are asserted in the flow
tests). `⚠n` = known quirk, listed below the chart.

```mermaid
flowchart TD
    classDef done fill:#d4efd4,stroke:#2e7d32,color:#14401b
    classDef err fill:#f9d6d5,stroke:#c0392b,color:#5c1410
    classDef quirk fill:#fff3cd,stroke:#b8860b,color:#5c4400

    subgraph ENTRY["Entry points"]
        HK1["Ctrl+Super+P<br/>(GNOME keybinding)"]
        HK2["Ctrl+Shift+Super+P"]
        TRAY["Tray menu"]
        ONESHOT["aiproof trigger --oneshot<br/>⚠1 ignores daemon 'enabled' state"]:::quirk
        STDIN["aiproof proofread (stdin)"]
        NAUT["Nautilus: Scripts →<br/>Proofread with AI"]
    end

    HK1 -->|"gdbus Trigger"| DBUS["daemon DBus dispatch<br/>(worker thread)"]
    HK2 -->|"gdbus TriggerClipboardOnly"| DBUS
    TRAY --> DBUS
    DBUS --> EN
    ONESHOT --> EN
    NAUT --> STDIN

    subgraph GATE["Gate (_begin + mode)"]
        EN{"enabled?"}
        LOCK{"lock free?"}
        MODE{"mode?"}
    end
    EN -->|"no"| NDIS["🔔 aiproof is disabled"]:::err
    EN -->|"yes"| LOCK
    LOCK -->|"no"| NBUSY["🔔 Already proofreading"]:::err
    LOCK -->|"yes"| MODE
    MODE -->|"paste_mode=clipboard-only<br/>(chosen: no ydotool note)"| PRIM
    MODE -->|"ydotool unavailable<br/>(degraded: note shown)"| PRIM
    MODE -->|"auto + ydotool"| SAVE

    subgraph PF["Paste flow (in place)"]
        SAVE["save clipboard<br/>(text or image)"] --> CLR["clear clipboard"]
        CLR --> CC["release modifiers<br/>+ synthetic Ctrl+C"]
        CC --> POLL{"clipboard content<br/>within 1.5s?"}
        POLL -->|"empty: retry<br/>Ctrl+C once"| POLL
        POLL -->|"still empty /<br/>whitespace-only"| PR1["restore clipboard"]
        POLL -->|"non-text"| PR2["restore clipboard"]
        POLL -->|"text"| PROOF1["proofread<br/>(LLM pipeline ↓)"]
        PROOF1 -->|"error message"| PR3["restore clipboard"]
        PROOF1 -->|"unchanged"| PR4["restore clipboard"]
        PROOF1 -->|"corrected"| SETC["set clipboard = correction"]
        SETC --> CV["release modifiers + Ctrl+V"]
        CV --> RST["wait restore_delay_ms,<br/>restore old clipboard"]
        XC["any crash → restore<br/>(if saved) → 🔔 failed"]:::err
    end
    PR1 --> NSEL["🔔 Select some text first"]:::err
    PR2 --> NNT["🔔 Selection is not text"]:::err
    PR3 --> NF1["🔔 Proofreading failed"]:::err
    RST --> NOK["🔔 Text corrected<br/>(+path note)"]:::done

    subgraph CF["Clipboard flow"]
        PRIM["read primary selection"] --> PEMPTY{"empty /<br/>whitespace-only?"}
        PEMPTY -->|"no"| PROOF2["proofread<br/>(LLM pipeline ↓)"]
        PROOF2 -->|"corrected"| SETC2["copy correction<br/>to clipboard"]
    end
    PEMPTY -->|"yes"| NSEL2["🔔 Select some text first"]:::err
    PROOF2 -->|"error message"| NF2["🔔 Proofreading failed"]:::err
    SETC2 --> NCP["🔔 Corrected in Xs (+path note)<br/>(+ydotool note if degraded)"]:::done

    PR4 --> UNCH
    PROOF2 -->|"unchanged"| UNCH
    UNCH{"last_path = fallback?<br/>(guards rejected everything)"}
    UNCH -->|"yes"| NUNSAFE["🔔 Couldn't apply<br/>corrections safely"]:::err
    UNCH -->|"no"| NCLEAN["🔔 No changes needed<br/>(+via-fallback note)"]:::done

    PROOF1 -.-> TL
    PROOF2 -.-> TL

    subgraph LLM["LLM pipeline (per provider attempt)"]
        TL{"len > max_chars?"}
        Q["query provider"]
        FC["full_clean"]
        WO{"empty or<br/>boilerplate-only?"}
        V{"guards pass?<br/>(layout, punctuation,<br/>appended content)"}
        CR["constrained retry<br/>against the ORIGINAL"]
        DR["repair prompt<br/>against the draft"]
        RV{"retry valid?"}
        PFB["path=fallback<br/>(original kept)"]
        LCNT{"line count?"}
        SEG["segment salvage"]
        LBL["line-by-line salvage"]
        SCH{"salvage changed<br/>anything?"}
        XESC["⚠2 non-LLMError transport errors<br/>(bad JSON, SSL) escape:<br/>'Unexpected error', no failover"]:::quirk
    end
    TL -->|"yes"| ETL["TextTooLongError<br/>(never fails over)"]:::err
    TL -->|"no"| Q
    Q --> FC --> WO
    WO -->|"yes"| ELLM["LLMError"]:::err
    WO -->|"no"| V
    V -->|"ok"| POK["path=ok"]:::done
    V -->|"punctuation only"| CR
    V -->|"layout broke"| DR
    CR --> RV
    DR --> RV
    RV -->|"yes"| PREP["path=repaired"]:::done
    RV -->|"no"| PFB
    RV -.->|"retry call raises LLMError:<br/>propagates (failover eligible)"| ELLM
    PFB --> LCNT
    LCNT -->|"1"| SEG
    LCNT -->|"2–40"| LBL
    LCNT -->|"over 40"| KEEP["keep original<br/>path=fallback"]:::err
    SEG --> SCH
    LBL --> SCH
    SCH -->|"yes"| PSAL["path=segments /<br/>line-by-line"]:::done
    SCH -->|"no"| KEEP

    subgraph FO["Provider failover"]
        FBC{"fallback provider<br/>configured?"}
        NFB["🔔 Primary provider unavailable —<br/>trying fallback…"]
        FBTRY["retry once with fallback<br/>provider/endpoint/model"]
        XPROV["⚠3 unknown provider id<br/>(ValueError) escapes<br/>before failover"]:::quirk
    end
    ELLM --> FBC
    ETL -.-> ERAISE
    FBC -->|"no"| ERAISE["error to caller"]:::err
    FBC -->|"yes"| NFB
    NFB --> FBTRY
    FBTRY -->|"succeeds"| FBOK["result + used_fallback<br/>('via fallback' note)"]:::done
    FBTRY -->|"fails too (logged)"| ERAISE

    subgraph CLI["aiproof proofread (stdin → stdout)"]
        SIN{"stdin empty?"}
        STRIP["strip single trailing newline<br/>(re-added on stdout)"]
        CRES{"result?"}
    end
    STDIN --> SIN
    SIN -->|"yes"| CEX1["exit 1"]:::err
    SIN -->|"no"| STRIP
    STRIP -.-> TL
    STRIP --> CRES
    CRES -->|"LLMError"| CEX2["exit 1 + message"]:::err
    CRES -->|"path=fallback"| CWARN["⚠4 prints NOT-verified warning<br/>but still exits 0<br/>(Nautilus reads only the exit code)"]:::quirk
    CRES -->|"ok"| COUT["stdout + [provider/model,<br/>path=…] + exit 0"]:::done
```

### Known quirks

Characterized (not fixed) behaviors, each pinned by a test so a change shows
up as a test failure:

1. **Oneshot ignores the daemon's enabled state** — `aiproof trigger
   --oneshot` builds a fresh in-process orchestrator with `enabled=True`
   (`test_trigger_oneshot`).
2. **Non-LLMError transport errors bypass failover** — a malformed JSON
   response, SSL error, etc. escapes `query()` unwrapped, so the fallback
   provider is never tried and the user sees "Unexpected error"
   (`test_query_other_requests_errors_escape`, `test_unexpected_error`).
3. **An unknown primary provider id never falls back** — the primary client
   is built outside the failover try
   (`test_unknown_primary_provider_never_falls_back`). Related: an exception
   raised inside the `on_fallback` callback aborts the retry
   (`test_on_fallback_exception_aborts_retry`).
4. **`aiproof proofread` exits 0 when the guards rejected everything**
   (`path=fallback`): the warning goes to stderr, which the Nautilus script
   discards, so a file run can report "Proofread" on unverified output
   (`test_proofread_guards_rejected_warns_but_exit_zero`).

### What the flow tests do not cover

Daemon lifecycle (bus-name ownership, config-file watcher, startup checks,
state marshalling), tray menu, hotkey registration, and the settings UI are
exercised only manually — they are thin GTK/GLib glue around the routes
charted above.
