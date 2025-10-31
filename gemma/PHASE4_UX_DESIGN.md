# Phase 4 UX Design: Model and Profile Management

## Design Philosophy

**Principles**:
- **Progressive Disclosure**: Show essential info first, details on demand
- **Visual Hierarchy**: Use color and typography to guide attention
- **Contextual Help**: Inline tips and examples where users need them
- **Forgiving Design**: Easy to undo, clear error messages, safe defaults
- **Accessibility**: Color is supplemental, not required for understanding

**Color Palette** (following existing Rich conventions):
- `cyan` - Primary information, headers, prompts
- `green` - Success states, recommended actions, active items
- `yellow` - Warnings, cautions, intermediate states
- `red` - Errors, destructive actions, unavailable items
- `dim` - Secondary info, hints, metadata
- `bold` - Emphasis, current selection, important values

---

## 1. Model Selection Flow

### 1.1 Model Discovery - List View

**When**: User runs `gemma-cli --list-models` or `/models` command during chat

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                        Available Gemma Models                               │
╰─────────────────────────────────────────────────────────────────────────────╯

┌────────┬──────────────────────┬────────┬──────────┬─────────┬──────────────┐
│ Status │ Model                │ Size   │ Speed    │ Quality │ Format       │
├────────┼──────────────────────┼────────┼──────────┼─────────┼──────────────┤
│   ✓    │ gemma-2b-it          │ 2.5 GB │ ████████ │ ████    │ SBS (single) │
│   ⚠    │ gemma-4b-it-sfp      │ 4.8 GB │ ████████ │ ██████  │ SFP          │
│   ✗    │ gemma-7b-it          │ 8.5 GB │ ████     │ ████████│ Not found    │
│        │ gemma-27b-it         │ 27 GB  │ ██       │ █████████│ Not found   │
└────────┴──────────────────────┴────────┴──────────┴─────────┴──────────────┘

Legend:
  ✓ = Available and verified    ████████ = Fast (recommended)
  ⚠ = Found but needs validation ████     = Moderate
  ✗ = Not found in model path    ██       = Slow

Current Model: gemma-2b-it (active)
Model Path:    C:\codedev\llm\.models\

Commands:
  gemma-cli --model=gemma-4b-it-sfp          Switch to different model
  gemma-cli --download gemma-7b-it            Download missing model
  gemma-cli --verify gemma-4b-it-sfp          Verify model integrity
```

**Color Coding**:
- Status: `green` (✓), `yellow` (⚠), `red` (✗)
- Model name: `cyan` for current, `dim` for unavailable
- Speed/Quality bars: `green` for high, `yellow` for medium, `red` for low
- Commands: `cyan` with `dim` for syntax

**Interaction Patterns**:
- Up/Down arrows to navigate (if interactive)
- Enter to select and switch
- ESC to cancel
- `?` for detailed model info

---

### 1.2 Model Comparison View

**When**: User runs `gemma-cli --compare-models` or selects multiple models

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                         Model Comparison                                     │
╰─────────────────────────────────────────────────────────────────────────────╯

                  ┌─────────────┬──────────────┬──────────────┐
                  │  gemma-2b   │  gemma-4b    │  gemma-7b    │
                  │     (active)│              │              │
┌─────────────────┼─────────────┼──────────────┼──────────────┤
│ File Size       │    2.5 GB   │    4.8 GB    │    8.5 GB    │
│ Memory Usage    │    ~4 GB    │    ~7 GB     │    ~12 GB    │
│ Tokens/sec      │    ~45      │    ~32       │    ~18       │
│ Context Window  │    8192     │    8192      │    8192      │
│ Quality Score   │    ████     │    ██████    │    ████████  │
│ Best For        │  Quick chat │  Balanced    │  Complex     │
│                 │  Testing    │  General use │  Reasoning   │
│ Status          │      ✓      │      ⚠       │      ✗       │
└─────────────────┴─────────────┴──────────────┴──────────────┘

System Recommendation: gemma-4b (balanced performance for your hardware)

Hardware Context:
  • Available RAM: 16 GB
  • CPU Cores:     8 cores (AVX2 support)
  • GPU:           None detected

Tip: gemma-4b-sfp format provides best speed/quality tradeoff

Switch to model:
  [1] gemma-2b-it    [2] gemma-4b-it-sfp    [3] Download gemma-7b-it
```

**Color Coding**:
- Active model column: `cyan` border
- Recommended model: `green` highlight
- Status icons: Same as list view
- "System Recommendation": `green` text on `dim` background

**Interaction**:
- Number keys (1-3) to select
- Left/Right arrows to navigate columns
- `d` to see detailed benchmarks
- `h` to show hardware requirements

---

### 1.3 Model Quick-Switch (In-Chat)

**When**: User types `/model` during active conversation

```
┌─ Model Quick Switch ─────────────────────────────────────────┐
│                                                               │
│  Current: gemma-2b-it  [████████████████] 2.5 GB            │
│                                                               │
│  Available Models:                                            │
│    1. gemma-2b-it        (current)  ████████  ████           │
│    2. gemma-4b-it-sfp    (recommended) ████████  ██████      │
│                                                               │
│  [⏎ Enter number] [ESC Cancel] [? More info]                 │
└───────────────────────────────────────────────────────────────┘

> /model 2

Switching to gemma-4b-it-sfp...
[████████████████████████████████████] 100%

✓ Model loaded successfully!
  Context preserved: 3 messages (512 tokens)
  Performance: ~32 tokens/sec (estimated)

Ready to continue conversation with improved model.
```

**Color Coding**:
- Panel border: `cyan`
- Current model: `green` with `bold`
- Recommended tag: `green` on `dim` background
- Progress bar: `cyan` → `green` when complete
- Success message: `green`

**Error States**:
```
✗ Model switch failed: gemma-4b-it-sfp not found

Suggestions:
  • Download model: gemma-cli --download gemma-4b-it-sfp
  • Verify model path: C:\codedev\llm\.models\
  • Use available model: /model 1 (gemma-2b-it)

Staying on gemma-2b-it (no changes made)
```

**Color**: `red` for error, `yellow` for suggestions, `dim` for path

---

### 1.4 Model Validation Feedback

**When**: Loading or switching models

```
Validating gemma-4b-it-sfp...

  ✓ Model file found      C:\...\4b-it-sfp.sbs  (4.8 GB)
  ✓ Tokenizer found       C:\...\tokenizer.spm  (512 KB)
  ⚙ Loading weights...    [████████████████] 100%
  ⚙ Initializing KV cache... [████████████] 100%
  ✓ Model ready

Performance Estimate:
  Tokens/sec:    ~32 (based on CPU: Intel i7-9700K)
  Memory usage:  ~7 GB (6.2 GB model + 0.8 GB cache)
  Context:       8192 tokens maximum

Tip: First query may be slower (warmup), subsequent faster
```

**Progressive States**:
- Pending: `⚙` with `cyan` spinner
- Complete: `✓` with `green` checkmark
- Failed: `✗` with `red` X

**Error Example**:
```
Validating gemma-4b-it-sfp...

  ✓ Model file found      C:\...\4b-it-sfp.sbs
  ✗ Tokenizer not found   Expected: C:\...\tokenizer.spm

Error: Cannot load model without tokenizer

Solutions:
  1. Download complete model package (includes tokenizer)
  2. Copy tokenizer from another model directory
  3. Specify tokenizer path: --tokenizer /path/to/tokenizer.spm

Need help? Run: gemma-cli --help model-setup
```

---

## 2. Profile Management Flow

### 2.1 Profile List View

**When**: User runs `gemma-cli --list-profiles` or `/profiles` command

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                      Performance Profiles                                    │
╰─────────────────────────────────────────────────────────────────────────────╯

┌─────────┬──────────────┬──────────────────────────┬──────────┬─────────────┐
│ Active  │ Profile      │ Description              │ Model    │ Performance │
├─────────┼──────────────┼──────────────────────────┼──────────┼─────────────┤
│    ✓    │ balanced     │ Best speed/quality mix   │ 2b/4b    │ ████████    │
│         │ speed        │ Fastest inference        │ 2b       │ ██████████  │
│         │ quality      │ Best output quality      │ 7b+      │ ████        │
│         │ memory-opt   │ Low memory footprint     │ 2b       │ ████████    │
│         │ creative     │ High temperature/variety │ Any      │ Variable    │
│    ⚡   │ auto-optimal │ Hardware-optimized       │ Auto     │ Adaptive    │
└─────────┴──────────────┴──────────────────────────┴──────────┴─────────────┘

Legend:
  ✓ = Currently active profile
  ⚡ = Recommended for your system

Current Settings:
  Profile:        balanced
  Model:          gemma-2b-it
  Threads:        8 (matched to CPU cores)
  Temperature:    0.7
  Max tokens:     2048

Commands:
  /profile speed              Switch to speed profile
  /profile new my-profile     Create custom profile
  /profile edit balanced      Modify existing profile
```

**Color Coding**:
- Active profile: `green` checkmark, row highlighted in `green dim`
- Recommended: `yellow` lightning bolt
- Profile names: `cyan`
- Performance bars: `green` (high) → `yellow` (medium) → `red` (low)

---

### 2.2 Profile Detailed View

**When**: User runs `gemma-cli --profile-info balanced`

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                  Profile: balanced (active)                                  │
╰─────────────────────────────────────────────────────────────────────────────╯

Description:
  Balanced performance profile optimized for general-purpose chat with
  good response quality and reasonable speed. Recommended for most users.

Configuration:
  ┌─────────────────────┬──────────────────────────────────────────┐
  │ Model Selection     │ gemma-2b-it or gemma-4b-it (auto)        │
  │ Thread Count        │ 8 (all available cores)                  │
  │ Context Window      │ 4096 tokens                              │
  │ Temperature         │ 0.7 (balanced creativity/consistency)    │
  │ Top-K               │ 40                                       │
  │ Top-P               │ 0.9                                      │
  │ Max Output Tokens   │ 2048                                     │
  │ KV Cache Mode       │ Dynamic compression                      │
  └─────────────────────┴──────────────────────────────────────────┘

Performance Characteristics:
  • Speed:          ~40-45 tokens/sec (gemma-2b)
  • Memory:         ~4-7 GB depending on model
  • Context length: Good (4K tokens)
  • Quality:        Good (suitable for most tasks)

Best Used For:
  ✓ General conversation
  ✓ Quick Q&A
  ✓ Code assistance
  ✓ Document summarization

Not Ideal For:
  ✗ Long-form creative writing (use 'creative' profile)
  ✗ Complex reasoning (use 'quality' profile)
  ✗ Low-memory systems (use 'memory-opt' profile)

Actions:
  [1] Activate this profile    [2] Edit settings    [3] Clone as new profile
```

**Color Coding**:
- Profile name: `cyan bold`
- Active indicator: `green`
- Settings table: `cyan` headers, `dim` values
- Checkmarks/X marks: `green` / `red`
- Actions: `cyan` with number shortcuts

---

### 2.3 Profile Creation Wizard

**When**: User runs `gemma-cli --profile-create` or `/profile new`

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                    Create New Performance Profile                            │
╰─────────────────────────────────────────────────────────────────────────────╯

Step 1 of 5: Profile Name

What would you like to call this profile?
> coding-assistant

Good choice! This name is available.

─────────────────────────────────────────────────────────────────────────────

Step 2 of 5: Primary Use Case

What will you primarily use this profile for?

  1. General conversation (balanced)
  2. Code generation and review
  3. Creative writing
  4. Question answering (speed-focused)
  5. Complex reasoning (quality-focused)
  6. Custom (I'll set everything manually)

> 2

Selected: Code generation and review

We'll optimize for:
  • Moderate temperature (consistent but not rigid)
  • Larger context window (whole files)
  • Faster model (quick iterations)

─────────────────────────────────────────────────────────────────────────────

Step 3 of 5: Model Selection

Based on your use case and hardware, we recommend:

  Recommended:  gemma-4b-it-sfp
  Why:          Good code quality, fast enough for iteration

  Alternatives:
    [1] gemma-2b-it       (faster, slightly lower quality)
    [2] gemma-7b-it       (higher quality, requires more memory)
    [3] Auto-select       (let system choose based on load)

Select model: [Enter for recommended] [1-3 for alternatives]
> ⏎

Using gemma-4b-it-sfp (recommended)

─────────────────────────────────────────────────────────────────────────────

Step 4 of 5: Fine-Tuning Parameters

Review and adjust settings: (Enter to keep default)

  Temperature:       0.6  (0.0 = deterministic, 1.0 = creative)
  Context Window:    6144 tokens (larger for code files)
  Max Output:        3072 tokens (longer code blocks)
  Thread Count:      8 (all available CPU cores)

Adjust temperature? [0.0-1.0] (current: 0.6)
> ⏎

Adjust context window? [1024-8192] (current: 6144)
> ⏎

─────────────────────────────────────────────────────────────────────────────

Step 5 of 5: Review and Confirm

Profile Summary:
  ┌──────────────────────┬────────────────────────────────────────┐
  │ Name                 │ coding-assistant                       │
  │ Purpose              │ Code generation and review             │
  │ Model                │ gemma-4b-it-sfp                        │
  │ Temperature          │ 0.6                                    │
  │ Context Window       │ 6144 tokens                            │
  │ Max Output           │ 3072 tokens                            │
  │ Threads              │ 8                                      │
  │ Estimated Speed      │ ~32 tokens/sec                         │
  │ Memory Usage         │ ~7 GB                                  │
  └──────────────────────┴────────────────────────────────────────┘

Save this profile? [Y/n]
> y

✓ Profile 'coding-assistant' created successfully!

Activate now? [Y/n]
> y

✓ Switched to 'coding-assistant' profile
  Model gemma-4b-it-sfp is now loading...

Ready to start coding session!
```

**Interaction Design**:
- **Step-by-step**: One question at a time, no overwhelm
- **Smart defaults**: Recommended values pre-filled, just press Enter
- **Validation**: Immediate feedback on invalid inputs
- **Reversible**: Can go back with `b` or cancel with `Ctrl+C`
- **Explain why**: Show reasoning for recommendations

**Color Coding**:
- Step headers: `cyan bold`
- Current input: `green` prompt
- Recommendations: `green` with "Recommended" tag
- Validation success: `green` checkmark
- Validation failure: `red` with inline error message

---

### 2.4 Profile Switching (In-Chat)

**When**: User types `/profile speed` during conversation

```
Current profile: balanced

Switch to profile: speed

╭─ Switching Profile ──────────────────────────────────────────╮
│                                                               │
│  From: balanced                                               │
│    • Model:       gemma-2b-it                                │
│    • Temperature: 0.7                                        │
│    • Context:     4096 tokens                                │
│                                                               │
│  To: speed                                                    │
│    • Model:       gemma-2b-it (no change)                    │
│    • Temperature: 0.5 (more deterministic)                   │
│    • Context:     2048 tokens (faster)                       │
│                                                               │
│  Impact:                                                      │
│    ⚡ 25% faster responses                                    │
│    ⚠  Slightly less creative outputs                         │
│    ⚠  Shorter context window (older messages truncated)      │
│                                                               │
│  Proceed? [Y/n]                                              │
└───────────────────────────────────────────────────────────────┘

> y

Applying profile 'speed'...
[████████████████████████████████████] 100%

✓ Profile activated!
  Context preserved: 2 most recent messages (512 tokens fit)
  1 older message truncated (to fit 2048 token limit)

Ready to continue with speed-optimized settings.
```

**Color Coding**:
- Panel border: `cyan`
- "From" section: `dim`
- "To" section: `green`
- Impact items: `yellow` for warnings, `green` for benefits
- Progress: `cyan` → `green`

**No-op Example** (already on target profile):
```
> /profile balanced

You're already using the 'balanced' profile. No changes made.

Tip: See other profiles with /profiles
```

---

### 2.5 Profile Edit Mode

**When**: User runs `gemma-cli --profile-edit balanced`

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                      Edit Profile: balanced                                  │
╰─────────────────────────────────────────────────────────────────────────────╯

Current Settings:
  [1] Model:           gemma-2b-it
  [2] Temperature:     0.7
  [3] Context Window:  4096 tokens
  [4] Max Output:      2048 tokens
  [5] Thread Count:    8
  [6] Top-K:           40
  [7] Top-P:           0.9
  [8] Advanced...      (more options)

Select setting to edit: [1-8] [S Save] [C Cancel]
> 2

Edit Temperature:
  Current:     0.7
  Range:       0.0 (deterministic) - 1.0 (creative)

  Common values:
    0.3-0.5    Very consistent, good for code/facts
    0.6-0.8    Balanced (default for most profiles)
    0.9-1.0    Creative, varied outputs

  New value: [0.0-1.0] (Enter to keep 0.7)
> 0.8

✓ Temperature updated: 0.7 → 0.8

Effect: Slightly more creative/varied responses

Return to settings? [Y/n]
> y

Current Settings:  (⚠ unsaved changes)
  [1] Model:           gemma-2b-it
  [2] Temperature:     0.8  ⚠ modified
  [3] Context Window:  4096 tokens
  ...

Select setting to edit: [1-8] [S Save] [C Cancel]
> S

Save changes to 'balanced' profile?
  • Temperature: 0.7 → 0.8

This will affect all future sessions using this profile.

Confirm? [Y/n]
> y

✓ Profile 'balanced' updated successfully!

Apply changes to current session? [Y/n]
> y

✓ Settings reloaded. New temperature active.
```

**Color Coding**:
- Modified values: `yellow` with ⚠ indicator
- Validation errors: `red` with inline help
- Save confirmation: `cyan` panel with `dim` details
- Success: `green` checkmarks

---

## 3. Hardware Detection UI

### 3.1 System Information Display

**When**: User runs `gemma-cli --system-info`

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                         System Information                                   │
╰─────────────────────────────────────────────────────────────────────────────╯

Hardware:
  ┌────────────────────┬─────────────────────────────────────────────┐
  │ CPU                │ Intel Core i7-9700K @ 3.60GHz              │
  │ CPU Cores          │ 8 physical, 8 logical                      │
  │ CPU Features       │ ✓ AVX2  ✓ FMA  ✓ SSE4.2                    │
  │ Total RAM          │ 16.0 GB (15.2 GB available)                │
  │ GPU                │ None detected (CPU inference only)         │
  │ Storage (models)   │ 487 GB free (C:\codedev\llm\.models\)     │
  └────────────────────┴─────────────────────────────────────────────┘

Software:
  ┌────────────────────┬─────────────────────────────────────────────┐
  │ OS                 │ Windows 10 Pro (22H2)                      │
  │ Compiler           │ MSVC 19.38 (Visual Studio 2022)            │
  │ CMake              │ 3.28.1                                     │
  │ Highway SIMD       │ 1.0.7 (AVX2 optimized)                     │
  └────────────────────┴─────────────────────────────────────────────┘

Model Compatibility:
  ✓ gemma-2b-it        Excellent (4 GB RAM, ~45 tokens/sec)
  ✓ gemma-4b-it-sfp    Good      (7 GB RAM, ~32 tokens/sec)
  ⚠ gemma-7b-it        Marginal  (12 GB RAM, ~18 tokens/sec, may swap)
  ✗ gemma-27b-it       Not viable (requires 32+ GB RAM)

Legend:
  ✓ = Recommended (good performance expected)
  ⚠ = Possible but may be slow or use swap memory
  ✗ = Not recommended (insufficient resources)

Performance Hints:
  • Your CPU supports AVX2 - inference is optimized!
  • 16 GB RAM allows comfortable use of 2B and 4B models
  • Consider upgrading to 32 GB RAM for 7B+ models
  • No GPU detected - CPU inference only (still fast with AVX2)
```

**Color Coding**:
- Section headers: `cyan bold`
- Table headers: `cyan`
- Feature checkmarks: `green` (✓), `red` (✗)
- Model compatibility: `green` (✓), `yellow` (⚠), `red` (✗)
- Estimates: `dim`
- Hints: `cyan` with `dim` bullet points

---

### 3.2 Recommendation Cards

**When**: User runs `gemma-cli --recommend` or first-time setup

```
╭─────────────────────────────────────────────────────────────────────────────╮
│              Personalized Recommendations for Your System                    │
╰─────────────────────────────────────────────────────────────────────────────╯

Based on your hardware:
  • Intel i7-9700K (8 cores, AVX2 support)
  • 16 GB RAM
  • No GPU

╭─────────────────────────────────────────────────────────────────────────────╮
│ 🌟 Best Overall: gemma-4b-it-sfp + balanced profile                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Why this combo?                                                             │
│   • Sweet spot for your 16 GB RAM                                           │
│   • SFP format optimized for AVX2 CPUs (2x faster)                          │
│   • Good quality for general chat, code, Q&A                                │
│                                                                              │
│ Expected Performance:                                                        │
│   Speed:           ~32 tokens/sec                                           │
│   Memory:          ~7 GB (plenty of headroom)                               │
│   Quality:         ████████ 8/10                                            │
│                                                                              │
│ [1] Use this recommendation                                                 │
╰─────────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────────────────╮
│ ⚡ Fastest Option: gemma-2b-it + speed profile                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ When to choose this:                                                        │
│   • Need quick responses (testing, rapid iteration)                         │
│   • Don't need highest quality outputs                                      │
│   • Want minimal memory footprint                                           │
│                                                                              │
│ Expected Performance:                                                        │
│   Speed:           ~48 tokens/sec (1.5x faster than 4B)                     │
│   Memory:          ~4 GB                                                    │
│   Quality:         ████████ 6/10                                            │
│                                                                              │
│ [2] Use this recommendation                                                 │
╰─────────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────────────────╮
│ 🎯 Best Quality (Stretch): gemma-7b-it + quality profile                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ ⚠ Warning: This may be tight on your 16 GB system                           │
│                                                                              │
│ When to choose this:                                                        │
│   • Need highest quality reasoning                                          │
│   • Have no other memory-intensive apps running                             │
│   • Willing to accept slower speed                                          │
│                                                                              │
│ Expected Performance:                                                        │
│   Speed:           ~18 tokens/sec (slower)                                  │
│   Memory:          ~12 GB (leaves 4 GB for OS)                              │
│   Quality:         ████████ 9/10                                            │
│                                                                              │
│ [3] Use this recommendation (advanced)                                      │
╰─────────────────────────────────────────────────────────────────────────────╯

Select recommendation: [1-3] [C Custom setup] [S Skip]
> 1

✓ Applying recommendation: gemma-4b-it-sfp + balanced profile

Next steps:
  [1/3] Checking if gemma-4b-it-sfp is downloaded...
  [2/3] Loading model into memory...
  [3/3] Activating balanced profile...

Setup complete! Ready to chat.
```

**Card Design**:
- **Emphasized recommendation**: `green` border, star emoji
- **Alternative options**: `cyan` border
- **Warnings**: `yellow` border, ⚠ symbol
- **Performance bars**: Visual quality indicators
- **Action buttons**: Numbered for easy selection

**Color Coding**:
- Star/Lightning icons: `green` / `yellow`
- "Why this combo" text: `cyan`
- Performance values: `dim`
- Warnings: `yellow` background with `bold` text
- Selected option: `green` highlight

---

### 3.3 Auto-Optimize Button UX

**When**: User runs `gemma-cli --auto-optimize`

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                         Auto-Optimize Settings                               │
╰─────────────────────────────────────────────────────────────────────────────╯

Analyzing your system...
  [████████████████████████████████████] 100%

✓ Hardware analysis complete

Detected Configuration:
  • CPU:      Intel i7-9700K (8 cores, AVX2 ✓)
  • RAM:      16 GB available
  • Storage:  487 GB free
  • GPU:      None (CPU inference)

Optimization Plan:
  ┌──────────────────────┬─────────────┬─────────────┬───────────────┐
  │ Setting              │ Current     │ Optimized   │ Improvement   │
  ├──────────────────────┼─────────────┼─────────────┼───────────────┤
  │ Thread Count         │ 4           │ 8           │ +100% speed   │
  │ Model                │ gemma-2b    │ gemma-4b    │ +30% quality  │
  │ Profile              │ default     │ balanced    │ Optimized     │
  │ SIMD Optimization    │ SSE4        │ AVX2        │ +50% ops      │
  │ KV Cache Mode        │ Static      │ Dynamic     │ -30% memory   │
  └──────────────────────┴─────────────┴─────────────┴───────────────┘

Expected Results:
  • Speed:   ~28 tok/s → ~32 tok/s (+14%)
  • Memory:  ~5.5 GB → ~7 GB (well within limits)
  • Quality: Improved output accuracy

Apply these optimizations? [Y/n]
> y

Applying optimizations...
  ✓ CPU threads: 4 → 8
  ✓ SIMD mode: SSE4 → AVX2
  ✓ Profile: default → balanced
  ⚙ Loading gemma-4b-it-sfp... [████████████████] 100%
  ✓ KV cache: static → dynamic compression

✓ Optimization complete!

Performance test:
  Running quick benchmark... [████████████████] 100%

  Result: 32.4 tokens/sec (within expected range ✓)

Your system is now optimized! Enjoy faster, higher-quality inference.
```

**Color Coding**:
- Analysis spinner: `cyan` with animation
- Detection results: `green` checkmarks
- Table: `cyan` headers, `dim` current, `green` optimized
- Improvement percentages: `green` (positive)
- Progress steps: `cyan` → `green` when complete
- Final benchmark: `green` with checkmark

**Safety Features**:
- Shows before/after comparison
- Estimates impacts clearly
- Requires confirmation before applying
- Validates changes with benchmark
- Can rollback if performance degrades

---

### 3.4 Performance Comparison View

**When**: User runs `gemma-cli --benchmark-compare`

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                    Performance Comparison (Live System)                      │
╰─────────────────────────────────────────────────────────────────────────────╯

Running benchmarks on your hardware...
  [████████████████████████████████████] 100% (3/3 models tested)

Results (averaged over 10 runs):

Tokens per Second:
  gemma-2b-it         ████████████████████████████ 48.3 tok/s
  gemma-4b-it-sfp     ████████████████████ 32.1 tok/s  (recommended)
  gemma-7b-it         ██████████ 17.8 tok/s

Memory Usage:
  gemma-2b-it         ████ 4.1 GB
  gemma-4b-it-sfp     ███████ 7.3 GB  (recommended)
  gemma-7b-it         ████████████ 12.4 GB

Time to First Token (TTFT):
  gemma-2b-it         ██ 0.4s
  gemma-4b-it-sfp     ███ 0.6s  (recommended)
  gemma-7b-it         ██████ 1.2s

Quality Score (subjective, based on test prompts):
  gemma-2b-it         ████████ 7.5/10
  gemma-4b-it-sfp     ████████████ 8.8/10  (recommended)
  gemma-7b-it         ██████████████ 9.2/10

Best Overall: gemma-4b-it-sfp
  • Balanced performance for your 16 GB system
  • Good speed/quality tradeoff
  • Comfortable memory headroom

Fastest: gemma-2b-it
  • 50% faster than 4B model
  • Slightly lower quality
  • Use for speed-critical tasks

Best Quality: gemma-7b-it
  • Highest quality outputs
  • 3x slower than 2B
  • May cause memory pressure

Export results? [Y/n]
> y

✓ Benchmark saved to: benchmarks/comparison_2025-01-13.json

Use this data to inform your model selection!
```

**Chart Design**:
- **Horizontal bars**: Easy visual comparison
- **Color gradient**: `green` (best) → `yellow` (middle) → `red` (worst)
- **Recommended tag**: `green` highlight on best overall choice
- **Live data**: Actual measurements from user's system
- **Percentile context**: Show where user falls vs. typical performance

---

## 4. Integration Points

### 4.1 Onboarding Wizard (First Launch)

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                  Welcome to Gemma CLI! (First Time Setup)                    │
╰─────────────────────────────────────────────────────────────────────────────╯

Let's configure gemma-cli for your system.

Step 1/4: Detect Hardware
  Scanning system... [████████████████████████████████████] 100%

  ✓ CPU: Intel i7-9700K (8 cores, AVX2 support)
  ✓ RAM: 16 GB
  ⚠ GPU: None (CPU inference only)

  Your system is well-suited for gemma-2b and gemma-4b models!

─────────────────────────────────────────────────────────────────────────────

Step 2/4: Choose Initial Model

We recommend: gemma-4b-it-sfp
  • Good quality for most tasks
  • Fits comfortably in 16 GB RAM
  • Fast enough for interactive chat

Alternatives:
  [1] gemma-2b-it       (faster, lower quality)
  [2] gemma-7b-it       (higher quality, slower, tight on memory)

Select model: [Enter for recommended] [1-2 for alternatives]
> ⏎

✓ Selected: gemma-4b-it-sfp

Checking if model is downloaded...
  ⚠ Model not found in: C:\codedev\llm\.models\

Download now? [Y/n] (4.8 GB)
> y

Downloading gemma-4b-it-sfp...
  [████████████████████████████████████] 100%  4.8 GB / 4.8 GB

✓ Model downloaded successfully!

─────────────────────────────────────────────────────────────────────────────

Step 3/4: Choose Performance Profile

What's your primary use case?

  [1] General chat (balanced speed/quality)
  [2] Code assistance (optimized for programming)
  [3] Creative writing (high variety/creativity)
  [4] Quick Q&A (speed-focused)

> 1

✓ Selected: balanced profile

Settings:
  • Temperature:  0.7 (balanced)
  • Context:      4096 tokens
  • Threads:      8 (all CPU cores)

─────────────────────────────────────────────────────────────────────────────

Step 4/4: Final Configuration

Review your setup:
  ┌─────────────────┬────────────────────────────────┐
  │ Model           │ gemma-4b-it-sfp               │
  │ Profile         │ balanced                      │
  │ CPU Threads     │ 8                             │
  │ Expected Speed  │ ~32 tokens/sec                │
  │ Memory Usage    │ ~7 GB                         │
  └─────────────────┴────────────────────────────────┘

Save configuration? [Y/n]
> y

✓ Configuration saved!

Loading model... [████████████████████████████████████] 100%

✓ Setup complete! gemma-cli is ready to use.

Type /help to see available commands
Type /exit to quit

Ready for your first message!
>
```

**Design Principles**:
- **Fast for experts**: Can press Enter to accept all recommendations
- **Guided for novices**: Clear explanations at each step
- **Smart defaults**: Based on actual hardware detection
- **Immediate validation**: Download prompt if model missing
- **Clear progress**: Step indicators and progress bars

---

### 4.2 Status Bar (During Chat)

**Top Bar (Always Visible)**:
```
╭─────────────────────────────────────────────────────────────────────────────╮
│ gemma-cli  |  Model: gemma-4b-it-sfp  |  Profile: balanced  |  ⚡ 32.1 tok/s│
╰─────────────────────────────────────────────────────────────────────────────╯

User: What is the capital of France?
Assistant: [Generating response with streaming...]

╭─ Response ────────────────────────────────────────────────────────────╮
│                                                                       │
│ The capital of France is Paris. It has been the capital since...     │
│ [Text streams here with typing effect]                               │
│                                                                       │
╰───────────────────────────────────────────────────────────────────────╯

[████████████████████████████████████] 100%  Completed in 2.3s

 32.4 tokens/sec  |  Memory: 7.1 GB  |  Context: 152/4096 tokens
```

**Color Coding**:
- Model name: `cyan`
- Profile: `dim cyan`
- Speed indicator: `green` if > 25 tok/s, `yellow` if 10-25, `red` if < 10
- Lightning bolt: `yellow` for speed emphasis

**Interactive States**:
```
 Model loading...        Model: gemma-4b-it-sfp  Profile: balanced  Loading...
 Model ready            Model: gemma-4b-it-sfp  Profile: balanced  ⚡ 32.1 tok/s
 Model error            Model: gemma-4b-it-sfp  Profile: balanced  ✗ Error
```

---

### 4.3 In-Chat Commands

**Command Palette** (triggered by `/`):
```
> /

Available Commands:
  Model & Profile:
    /model [name]        Switch to different model
    /profile [name]      Switch performance profile
    /models              List all available models
    /profiles            List all performance profiles

  Conversation:
    /clear               Clear conversation history
    /save [name]         Save current conversation
    /load [name]         Load saved conversation
    /export [format]     Export chat (txt, md, json)

  Settings:
    /settings            Open settings menu
    /status              Show detailed system status
    /benchmark           Run performance benchmark

  Help:
    /help [command]      Show command help
    /exit                Exit gemma-cli

Type command name for autocomplete...
```

**Model Switch** (`/model`):
```
> /model

Current: gemma-4b-it-sfp

Available models:
  [1] gemma-2b-it       ✓ Loaded   2.5 GB   ████████  Fast
  [2] gemma-4b-it-sfp   ✓ Active   4.8 GB   ████████  Current
  [3] gemma-7b-it       ✗ Missing  8.5 GB   ████      Download

Switch to: [1-3] [ESC Cancel]
> 1

Switching to gemma-2b-it...
  ⚙ Unloading gemma-4b-it-sfp...
  ⚙ Loading gemma-2b-it...
  ✓ Model loaded (0.8s)

Context preserved: 3 messages
Ready to continue with faster model.
```

**Profile Switch** (`/profile`):
```
> /profile

Current: balanced

Available profiles:
  [1] speed         Fastest inference
  [2] balanced      Speed/quality mix (current)
  [3] quality       Best outputs
  [4] memory-opt    Low memory usage
  [5] creative      High variety

Switch to: [1-5] [ESC Cancel]
> 1

Applied 'speed' profile
  • Temperature: 0.7 → 0.5
  • Context: 4096 → 2048 tokens
  • Estimated speed: +25%
```

**Settings Menu** (`/settings`):
```
> /settings

╭─ Settings ────────────────────────────────────────────────────────────╮
│                                                                       │
│  [1] Model Settings                                                   │
│      Current: gemma-4b-it-sfp                                        │
│      ▸ Change model, download new models                             │
│                                                                       │
│  [2] Performance Profile                                              │
│      Current: balanced                                               │
│      ▸ Switch or create profiles                                     │
│                                                                       │
│  [3] Advanced Parameters                                              │
│      Temperature: 0.7  |  Top-P: 0.9  |  Max tokens: 2048           │
│      ▸ Fine-tune inference parameters                                │
│                                                                       │
│  [4] System Info                                                      │
│      CPU: 8 cores  |  RAM: 16 GB  |  GPU: None                      │
│      ▸ View detailed system information                              │
│                                                                       │
│  [5] Paths & Storage                                                  │
│      Models: C:\codedev\llm\.models\                                 │
│      ▸ Configure storage locations                                   │
│                                                                       │
│  [ESC] Back to chat                                                  │
│                                                                       │
╰───────────────────────────────────────────────────────────────────────╯

Select: [1-5] [ESC Cancel]
```

---

### 4.4 Settings Menu Structure

**Main Settings Screen**:
```
╭─────────────────────────────────────────────────────────────────────────────╮
│                             Settings                                         │
╰─────────────────────────────────────────────────────────────────────────────╯

Navigation: [←→ Tab between sections] [↑↓ Navigate items] [⏎ Select] [ESC Back]

┌─ Model ─────────────┬─ Performance ────────┬─ Advanced ───────────────────┐
│                     │                      │                              │
│ Current Model:      │ Active Profile:      │ Temperature:      0.7        │
│   gemma-4b-it-sfp   │   balanced           │ Top-P:            0.9        │
│                     │                      │ Top-K:            40         │
│ Available:          │ Available:           │ Max Tokens:       2048       │
│   ✓ 2b-it           │   • speed            │ Context Window:   4096       │
│   ✓ 4b-it-sfp ★     │   • balanced ★       │ Thread Count:     8          │
│   ✗ 7b-it           │   • quality          │ Seed:             Random     │
│                     │   • memory-opt       │                              │
│ [Change Model]      │   • creative         │ [Edit Parameters]            │
│ [Download Model]    │                      │ [Reset to Defaults]          │
│                     │ [Switch Profile]     │                              │
│                     │ [Create Profile]     │                              │
└─────────────────────┴──────────────────────┴──────────────────────────────┘

┌─ System ────────────┬─ Paths ──────────────┬─ About ──────────────────────┐
│                     │                      │                              │
│ CPU:    i7-9700K    │ Model Directory:     │ gemma-cli v1.0.0             │
│ Cores:  8           │   C:\codedev\llm\... │                              │
│ RAM:    16 GB       │                      │ Based on gemma.cpp           │
│ GPU:    None        │ Config File:         │ Build: 2025-01-13            │
│ SIMD:   AVX2 ✓      │   ~/.gemma-cli.json  │                              │
│                     │                      │ [Check Updates]              │
│ [Run Diagnostics]   │ [Change Paths]       │ [View License]               │
│ [View Benchmarks]   │ [Clear Cache]        │ [Documentation]              │
└─────────────────────┴──────────────────────┴──────────────────────────────┘

 [S] Save Changes    [R] Reset All    [ESC] Cancel
```

**Color Coding**:
- Section borders: `cyan`
- Active section: `green` border
- Selected item: `green` highlight
- Current/active indicator (★): `green`
- Unavailable items: `dim red`
- Action buttons: `cyan` with keyboard shortcuts in `yellow`

---

## 5. Error State Handling

### 5.1 Model Not Found

```
╭─ Error ───────────────────────────────────────────────────────────────╮
│                                                                       │
│  ✗ Model Not Found: gemma-4b-it-sfp                                  │
│                                                                       │
│  Expected location:                                                   │
│    C:\codedev\llm\.models\gemma-4b-it-sfp\4b-it-sfp.sbs             │
│                                                                       │
│  This usually means:                                                  │
│    • Model hasn't been downloaded yet                                │
│    • Model files were moved or deleted                               │
│    • Model path is incorrectly configured                            │
│                                                                       │
│  Solutions:                                                           │
│    [1] Download model now (4.8 GB)                                   │
│    [2] Use different model (gemma-2b-it available)                   │
│    [3] Specify custom model path                                     │
│    [4] View model directory contents                                 │
│                                                                       │
│  Select option: [1-4] [ESC Cancel]                                   │
│                                                                       │
╰───────────────────────────────────────────────────────────────────────╯
```

**Color**: `red` border, `yellow` bullet points for solutions

---

### 5.2 Insufficient Memory

```
╭─ Warning ─────────────────────────────────────────────────────────────╮
│                                                                       │
│  ⚠ Insufficient Memory for gemma-7b-it                               │
│                                                                       │
│  Required:      ~12 GB                                               │
│  Available:     8.2 GB (16 GB total, 7.8 GB in use)                  │
│  Shortfall:     ~4 GB                                                │
│                                                                       │
│  Risks if you proceed:                                                │
│    • System may use swap memory (very slow)                          │
│    • Other applications may crash                                    │
│    • Model loading may fail midway                                   │
│                                                                       │
│  Recommendations:                                                     │
│    [1] Use smaller model (gemma-4b-it: 7 GB, available)             │
│    [2] Close other applications and retry                            │
│    [3] Try 'memory-opt' profile (reduces usage ~20%)                 │
│    [4] Proceed anyway (not recommended)                              │
│                                                                       │
│  Select option: [1-4]                                                │
│                                                                       │
╰───────────────────────────────────────────────────────────────────────╯
```

**Color**: `yellow` border, `red` for risks, `green` for recommendations

---

### 5.3 Model Loading Failure

```
╭─ Error ───────────────────────────────────────────────────────────────╮
│                                                                       │
│  ✗ Failed to Load Model: gemma-4b-it-sfp                             │
│                                                                       │
│  Error details:                                                       │
│    Invalid weight format at offset 0x1A3F2000                        │
│    Expected magic number: 0x42534730                                 │
│    Found: 0x00000000                                                 │
│                                                                       │
│  Possible causes:                                                     │
│    • Model file is corrupted                                         │
│    • Incomplete download                                             │
│    • Wrong file format (expected .sbs)                               │
│    • Storage media error                                             │
│                                                                       │
│  Troubleshooting:                                                     │
│    [1] Re-download model (replaces corrupted file)                   │
│    [2] Verify file integrity (check MD5/SHA256)                      │
│    [3] Try different model format (BF16 instead of SFP)              │
│    [4] Check disk space and permissions                              │
│    [5] View detailed error log                                       │
│                                                                       │
│  Select option: [1-5] [C Continue with different model]              │
│                                                                       │
╰───────────────────────────────────────────────────────────────────────╯
```

**Color**: `red` border, `dim red` for error details, `cyan` for troubleshooting options

---

### 5.4 Profile Conflict

```
╭─ Conflict ────────────────────────────────────────────────────────────╮
│                                                                       │
│  ⚠ Profile/Model Mismatch                                            │
│                                                                       │
│  You're trying to use:                                                │
│    Profile:  quality (optimized for 7B+ models)                      │
│    Model:    gemma-2b-it (small model)                               │
│                                                                       │
│  This combination may result in:                                      │
│    • Slower inference than necessary                                 │
│    • Wasted memory allocation                                        │
│    • Suboptimal quality (model too small for profile goals)          │
│                                                                       │
│  Recommendations:                                                     │
│    [1] Use 'balanced' profile with gemma-2b-it (optimal)             │
│    [2] Switch to gemma-7b-it model (matches profile)                 │
│    [3] Continue anyway (not recommended)                             │
│                                                                       │
│  Auto-optimize? [Y/n] (applies recommendation #1)                     │
│                                                                       │
╰───────────────────────────────────────────────────────────────────────╯
```

**Color**: `yellow` border, `green` for auto-optimize prompt

---

### 5.5 Performance Degradation

```
╭─ Performance Alert ───────────────────────────────────────────────────╮
│                                                                       │
│  ⚠ Inference Speed Below Expected                                    │
│                                                                       │
│  Current:    18.3 tokens/sec                                         │
│  Expected:   ~32 tokens/sec (gemma-4b-it-sfp + balanced)             │
│  Deviation:  -43% slower than baseline                               │
│                                                                       │
│  Possible causes:                                                     │
│    • High CPU load from other applications                           │
│    • Thermal throttling (CPU overheating)                            │
│    • Memory pressure (swapping)                                      │
│    • Background system tasks                                         │
│    • Power-saving mode enabled                                       │
│                                                                       │
│  Diagnostics:                                                         │
│    CPU Usage:      87% (normal: <70%)                               │
│    Memory:         14.2 GB / 16 GB (tight)                           │
│    Swap Usage:     2.1 GB (indicates memory pressure)                │
│    CPU Frequency:  2.4 GHz (throttled from 3.6 GHz)                 │
│                                                                       │
│  Actions:                                                             │
│    [1] Switch to 'speed' profile (lighter weight)                    │
│    [2] Switch to gemma-2b-it (smaller model)                         │
│    [3] View top CPU consumers                                        │
│    [4] Pause and investigate                                         │
│    [5] Continue anyway                                               │
│                                                                       │
│  Select option: [1-5]                                                │
│                                                                       │
╰───────────────────────────────────────────────────────────────────────╯
```

**Color**: `yellow` border, `red` for high values, `green` for normal ranges

---

## 6. Accessibility Considerations

### 6.1 Screen Reader Support

**Announcements**:
- Model loading: "Loading gemma-4b-it-sfp model. Please wait."
- Progress: "Loading: 45 percent complete"
- Model ready: "Model loaded successfully. Ready for input."
- Streaming: "Assistant is generating response. Token 23 of estimated 150."
- Completion: "Response complete. 142 tokens in 4.2 seconds."

**Semantic Markup**:
- Use ARIA labels for progress bars
- Announce state changes immediately
- Provide text alternatives for all visual indicators
- Support keyboard navigation throughout

---

### 6.2 Keyboard Navigation

**Global Shortcuts**:
- `Ctrl+N`: New conversation
- `Ctrl+O`: Open saved conversation
- `Ctrl+S`: Save conversation
- `Ctrl+,`: Open settings
- `Ctrl+/`: Show command palette
- `Ctrl+Q`: Quit application
- `Ctrl+C`: Cancel current operation
- `F1`: Help

**In Menus**:
- `↑↓`: Navigate options
- `←→`: Switch tabs/sections
- `⏎`: Select/activate
- `ESC`: Go back/cancel
- `Tab`: Move to next control
- `Shift+Tab`: Move to previous control
- Numbers `1-9`: Quick select (where applicable)

---

### 6.3 High Contrast Mode

**Adaptations**:
- Increase color saturation by 30%
- Use bold text for important elements
- Add borders around all panels
- Increase font size by 10%
- Use patterns in addition to colors (bars have texture)

**Example**:
```
Normal:          ████████ 8/10
High Contrast:   ▓▓▓▓▓▓▓▓ 8/10 (bold, thicker bars)
```

---

### 6.4 Reduced Motion Mode

**Adaptations**:
- No animated progress bars (show percentage only)
- No streaming text effects (show complete text)
- No smooth scrolling
- No spinners (use static indicators)

**Example**:
```
Normal:          [████████████          ] 60% Loading...
Reduced Motion:  Loading: 60% complete
```

---

## 7. Implementation Notes

### 7.1 Layout System

Use Rich's layout primitives:
- `Panel` for bordered sections
- `Table` for tabular data
- `Tree` for hierarchical views
- `Progress` for loading states
- `Live` for streaming updates
- `Layout` for complex multi-pane views

---

### 7.2 Responsive Design

**Terminal Width Adaptations**:
- < 80 cols: Minimal UI, single column
- 80-120 cols: Standard UI (target)
- > 120 cols: Expanded UI with side panels

**Height Adaptations**:
- < 24 rows: Compact mode, minimal chrome
- 24-40 rows: Standard mode
- > 40 rows: Expanded mode with context preview

---

### 7.3 State Management

**Persistent State** (saved to config file):
- Active model
- Active profile
- User preferences
- Custom profiles
- Model download history

**Session State** (in memory):
- Current conversation
- Model load status
- Performance metrics
- Recent commands

---

### 7.4 Performance Considerations

**Rendering Optimization**:
- Use `Live` context for streaming to minimize redraws
- Batch panel updates
- Lazy-load model lists
- Cache formatted tables
- Debounce progress updates (max 10 updates/sec)

**Memory Management**:
- Limit console history to 1000 lines
- Truncate long messages in display
- Stream long outputs to temp file if > 10MB
- Clear old metrics after conversation end

---

## 8. Testing Checklist

### 8.1 Visual Testing

- [ ] All colors distinguishable in light/dark terminals
- [ ] Borders align properly at all terminal sizes
- [ ] Progress bars animate smoothly
- [ ] Tables don't overflow terminal width
- [ ] Status bar updates in real-time

### 8.2 Interaction Testing

- [ ] All keyboard shortcuts work
- [ ] Mouse clicks registered (if terminal supports)
- [ ] Tab navigation flows logically
- [ ] ESC cancels operations properly
- [ ] Ctrl+C interrupts long operations

### 8.3 Error State Testing

- [ ] Model not found shows helpful message
- [ ] Out of memory gracefully degrades
- [ ] Corrupted model shows recovery options
- [ ] Network errors show retry logic
- [ ] Invalid input shows validation message

### 8.4 Accessibility Testing

- [ ] Screen reader announces all states
- [ ] High contrast mode readable
- [ ] Reduced motion mode works
- [ ] Keyboard-only navigation possible
- [ ] All actions have text alternatives

---

## 9. Design Rationale Summary

### Why This Design?

1. **Progressive Disclosure**: Users see what they need when they need it
   - Beginners: Guided wizard with recommendations
   - Experts: Quick commands and keyboard shortcuts
   - Everyone: Contextual help inline

2. **Visual Hierarchy**: Clear information architecture
   - Most important info (current state) always visible in status bar
   - Secondary info (options) one command away
   - Detailed info (system specs) available on demand

3. **Consistency**: Patterns repeated throughout
   - Same color scheme everywhere
   - Same interaction patterns (numbers for selection, ESC to cancel)
   - Same error handling approach (show problem + solutions)

4. **Forgiving**: Easy to recover from mistakes
   - All actions confirmable
   - Changes shown before applying
   - Rollback options available
   - Clear cancel mechanisms

5. **Performance-Aware**: Design reflects system realities
   - Shows actual benchmarks from user's hardware
   - Warns about resource constraints
   - Recommends optimal configurations
   - Monitors performance in real-time

6. **Accessible by Default**: Inclusive from the start
   - Color supplements info, doesn't replace it
   - Keyboard navigation throughout
   - Screen reader support built-in
   - Reduced motion option available

---

## 10. Next Steps

### Implementation Phases

**Phase 1**: Core model management UI
- Model list view
- Model switching
- Validation feedback

**Phase 2**: Profile system UI
- Profile list view
- Profile creation wizard
- Profile switching

**Phase 3**: Hardware detection UI
- System info display
- Recommendation cards
- Auto-optimize feature

**Phase 4**: Settings and integration
- Settings menu
- Status bar
- In-chat commands
- Onboarding wizard

**Phase 5**: Polish and accessibility
- High contrast mode
- Reduced motion
- Screen reader testing
- Performance optimization

---

*End of UX Design Document*
