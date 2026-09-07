"""Generate SRT/VTT caption files from word-level timing data."""

from pathlib import Path

from .helpers import WordTiming

SENTENCE_END_CHARS = ".!?"


def _join_words(words: list[WordTiming]) -> str:
    """Join words into readable text, respecting original spacing.

    Kokoro's tokenizer emits punctuation as separate tokens (e.g. "sentence"
    then "."), so a naive space-joined string would read "sentence . next".
    Each word's `trailing_space` says whether a space follows it.
    """
    text = "".join(
        w.text + (" " if w.trailing_space else "") for w in words
    )
    return text.strip()


def group_words_into_cues(
    words: list[WordTiming],
    max_duration: float = 7.0,
    max_words: int = 14,
) -> list[tuple[float, float, str]]:
    """Group words into readable caption cues.

    Breaks preferentially at sentence-ending punctuation, falling back to
    a maximum duration or word count so long sentences still get split
    into short, readable lines.

    Args:
        words: Word timings in chronological order.
        max_duration: Maximum cue length in seconds before forcing a break.
        max_words: Maximum words per cue before forcing a break.

    Returns:
        list[tuple[float, float, str]]: (start, end, text) cues.
    """
    cues: list[tuple[float, float, str]] = []
    current: list[WordTiming] = []

    for word in words:
        current.append(word)
        duration = current[-1].end - current[0].start
        ends_sentence = word.text[-1:] in SENTENCE_END_CHARS
        if ends_sentence or len(current) >= max_words or duration >= max_duration:
            cues.append((current[0].start, current[-1].end, _join_words(current)))
            current = []

    if current:
        cues.append((current[0].start, current[-1].end, _join_words(current)))

    return cues


def _format_srt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    ms = round((s - int(s)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def _format_vtt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    ms = round((s - int(s)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"


def write_srt(cues: list[tuple[float, float, str]], path: Path) -> None:
    """Write cues to an SRT (SubRip) file."""
    with open(path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(cues, 1):
            f.write(
                f"{i}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n"
                f"{text}\n\n"
            )


def write_vtt(cues: list[tuple[float, float, str]], path: Path) -> None:
    """Write cues to a WebVTT file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for start, end, text in cues:
            f.write(f"{_format_vtt_time(start)} --> {_format_vtt_time(end)}\n{text}\n\n")
