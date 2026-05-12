# Lexer specification (v0)

Source of truth for the lexer: `docs/grammar/grammar.ebnf` (lexical
section) plus this document, which specifies character predicates and
disambiguation rules that EBNF can't comfortably express.

The lexer is hand-written recursive descent. Each token is emitted with
a `SourceSpan` (byte offsets + 1-based line/col of the start). The lexer
never throws — `Lexer::next()` returns `parse_expected<Token>`. End of
input is a `TokenKind::EndOfFile` value, not an error.

## Token kinds

| Kind | Lexeme(s) / regex | Notes |
|---|---|---|
| `Identifier` | `[A-Za-z_.][A-Za-z0-9_.]*` | The literal `NA` and `c` and `start` lex as `Identifier`; the parser interprets them in context. |
| `NumLit` | `[0-9]+ (\.[0-9]*)? ([eE][+-]?[0-9]+)?` &#124; `\.[0-9]+ ([eE][+-]?[0-9]+)?` | A leading sign is **never** part of the number — `-1` lexes as `Op('-')` then `NumLit(1)`. |
| `StringLit` | `"…"` | `\` is **not** an escape in v0 (lavaan treats backslashes literally inside `"…"`). A newline inside a string is an error. |
| `Op` | one of `=~`, `~~`, `~`, `:=`, `==`, `<`, `>`, `<~`, `\|~`, `~*~`, `\|` | See "Operator longest-match" below. The four `<~`, `\|~`, `~*~`, `\|` are lexed normally and the parser produces an `UnsupportedOperator` error. |
| `Plus` | `+` | Used both as RHS-term separator and as `expr` operator. The parser disambiguates by context. |
| `Minus` | `-` | Binary subtraction or unary negation, only inside `expr` (constraints / define-param). |
| `Star` | `*` | The modifier separator. Also binary multiplication inside `expr`. |
| `Slash` | `/` | Binary division, only inside `expr`. |
| `Caret` | `^` | Binary power, right-associative, only inside `expr`. |
| `Question` | `?` | Start-value modifier separator. |
| `Comma` | `,` | Inside `c(...)` and `start(...)`. |
| `LParen` / `RParen` | `(` / `)` | |
| `Semicolon` | `;` | Statement separator. |
| `Newline` | `\n` &#124; `\r\n` &#124; `\r` | Statement separator. Significant. |
| `EndOfFile` | — | Synthesized at end of input. |

## Whitespace and comments

- `WS` is `' '` or `'\t'`. It separates tokens but is otherwise discarded.
  Whitespace is **required** between two adjacent tokens that would
  otherwise concatenate (e.g. between two identifiers). Whitespace
  inside operators is rejected: `= ~` is a `ParseError::Kind::UnknownOperator`
  on the `=`.
- Comments start with `#` or `!` and run to the next newline (or EOF).
  Comments are stripped before any token is emitted; they do not produce
  a `Token`.
- Form-feed (`\f`) and vertical tab (`\v`) are not recognized in v0;
  encountering one is `ParseError::Kind::UnexpectedChar`.

## Newlines

- `\r\n` is one `Newline` token, not two.
- A bare `\r` (classic Mac line ending) is also one `Newline`. In practice
  this is rarely seen but it costs nothing to accept.
- Multiple consecutive newlines collapse into one `Newline` token at the
  parser level (the lexer emits each separately; the parser skips runs).

## Multi-line continuation

Lavaan implicitly continues a formula across newlines when a `+` is
floating at the end (or start) of a line:

```
f =~ x1 + x2 +
     x3 + x4
```

In our lexer, both `Plus` and `Newline` are emitted as separate tokens.
The parser's `rhs_list` rule treats a `Newline` followed (after optional
runs of `Newline`) by `Plus` as a continuation, and a `Plus` followed
by `Newline` likewise. There is no special lexer state.

## Operator longest-match

Operators are scanned left-to-right with longest-match. The order of
recognition (longest first) matters only because some operators share
prefixes:

```
'~*~'   (3 chars) — rejected
'<~'    (2 chars) — rejected
'|~'    (2 chars) — rejected
'~~'    (2 chars)
'=~'    (2 chars)
'=='    (2 chars)
':='    (2 chars)
'~'     (1 char)
'<'     (1 char)
'>'     (1 char)
'|'     (1 char) — rejected
```

A lone `=` is `ParseError::Kind::UnknownOperator` (we never see one
outside `=~` or `==`).

## Numeric literals

- Leading sign is **never** part of the number. Lexer never produces
  a negative `NumLit`.
- Leading `.` is allowed: `.5` is a valid `NumLit`.
- Trailing `.` is allowed: `5.` is a valid `NumLit`.
- Exponent: `1e3`, `1.5E-2`, `.5e+10` all valid.
- Hex, octal, binary literals are **not** recognized in v0.
- A bare `NA` is an `Identifier`, not a `NumLit`. The parser distinguishes.

## Identifiers

- `id_start = [A-Za-z_.]`, `id_cont = [A-Za-z0-9_.]`. A leading `.`
  is permitted (matches lavaan / R conventions for hidden-ish names).
- Identifiers are case-sensitive (lavaan is too).
- There are no reserved words at the lexer level. The parser treats
  `c`, `start`, and `NA` specially in modifier contexts.

## String literals

- Delimited by `"`. Single quotes are not string delimiters in v0.
- No escape sequences. `"a\nb"` is the literal four-character string
  `a\nb`. (Lavaan parses through R, which gives different semantics, but
  inside model strings backslashes are pass-through.)
- A newline inside a string is `ParseError::Kind::UnterminatedString`
  with a span pointing at the opening `"`.
- An EOF inside a string is the same error.

## Errors the lexer emits

| `ParseError::Kind` | When |
|---|---|
| `UnexpectedChar` | Any character outside the recognized set: control chars (except `\t`, `\n`, `\r`), non-ASCII bytes, etc. |
| `UnknownOperator` | A symbol or symbol cluster that does not match any operator (e.g. lone `=`, `&`, etc.). |
| `UnterminatedString` | EOF or newline inside `"…"`. |
| `MalformedNumber` | Number with `e` followed by no digits, or trailing `.` adjacent to an `id_start` character. |

`UnsupportedOperator` (`<~`, `\|~`, `~*~`, `\|`) is **not** emitted by
the lexer. The lexer emits these as `Op` tokens and the parser raises
the unsupported error so the diagnostic span covers the whole operator
in context.
