"""End-of-day CERT r4.2 aggregates; no fitted state or historical labels."""
from collections import Counter, defaultdict
from datetime import date
from urllib.parse import urlsplit

FEATURE_SET = "cert_account_day_raw_v1"
SOURCES = ("logon", "device", "file", "http", "email")
HEADERS = {
    "logon": ["id", "date", "user", "pc", "activity"],
    "device": ["id", "date", "user", "pc", "activity"],
    "file": ["id", "date", "user", "pc", "filename", "content"],
    "http": ["id", "date", "user", "pc", "url", "content"],
    "email": ["id", "date", "user", "pc", "to", "cc", "bcc", "from", "size", "attachments", "content"],
}
EXTENSIONS = {
    "document": {"doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "rtf", "odt"},
    "pdf": {"pdf"}, "archive": {"zip", "rar", "7z", "gz", "tar"},
    "image": {"jpg", "jpeg", "png", "gif", "bmp", "tif", "tiff"},
    "executable": {"exe", "dll", "bat", "cmd", "ps1", "sh"},
}
BASE_FEATURES = [
    "total_events", "active_sources", "is_weekend", "distinct_pcs", "off_hours_events",
    "first_second", "last_second", "activity_span_seconds", "gap_count",
    "gap_mean_seconds", "gap_std_seconds", "gap_max_seconds",
    "logon_count", "logoff_count", "connect_count", "disconnect_count",
    "device_matched_pairs", "device_paired_seconds", "device_unmatched_connects",
    "device_unmatched_disconnects", "file_distinct_names",
    *[f"file_extension_{k}_count" for k in (*EXTENSIONS, "other")],
    "http_distinct_domains", "http_distinct_urls", "http_invalid_url_count",
    "email_size_bytes_sum", "email_size_bytes_max", "email_size_bytes_mean",
    "email_attachment_count", "email_with_attachment_count",
    "email_internal_recipient_count", "email_external_recipient_count",
    "email_distinct_recipients", "email_external_message_count",
]
FEATURES = BASE_FEATURES + [f"{s}_{suffix}" for s in SOURCES
                           for suffix in ("events", "off_hours_events", "distinct_pcs", "fraction")]


class Day:
    def __init__(self):
        self.counts = Counter()
        self.sets = defaultdict(set)
        self.first = self.last = None
        self.gap_mean = self.gap_m2 = self.gap_max = 0.0
        self.gaps = 0
        self.connections = {}
        self.label = 0

    def add(self, source, row, second, label):
        c, sets = self.counts, self.sets
        if self.last is not None:
            gap = second - self.last
            if gap < 0:
                raise ValueError("Account-day events are not chronological")
            self.gaps += 1
            delta = gap - self.gap_mean
            self.gap_mean += delta / self.gaps
            self.gap_m2 += delta * (gap - self.gap_mean)
            self.gap_max = max(self.gap_max, gap)
        else:
            self.first = second
        self.last = second
        self.label = max(self.label, label)
        off = second < 7 * 3600 or second >= 19 * 3600
        c["total_events"] += 1
        c[f"{source}_events"] += 1
        c["off_hours_events"] += off
        c[f"{source}_off_hours_events"] += off
        sets["pcs"].add(row[3])
        sets[f"{source}_pcs"].add(row[3])
        if source == "logon":
            if row[4] not in {"Logon", "Logoff"}:
                raise ValueError("Unknown logon action")
            c[row[4].lower() + "_count"] += 1
        elif source == "device":
            pc, action = row[3], row[4]
            if action == "Connect":
                c["connect_count"] += 1
                c["device_unmatched_connects"] += int(pc in self.connections)
                self.connections[pc] = second
            elif action == "Disconnect":
                c["disconnect_count"] += 1
                start = self.connections.pop(pc, None)
                if start is None:
                    c["device_unmatched_disconnects"] += 1
                else:
                    c["device_matched_pairs"] += 1
                    c["device_paired_seconds"] += second - start
            else:
                raise ValueError("Unknown device action")
        elif source == "file":
            if not row[4]:
                raise ValueError("Empty filename")
            sets["files"].add(row[4])
            extension = row[4].replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[-1].lower()
            bucket = next((k for k, values in EXTENSIONS.items() if extension in values), "other")
            c[f"file_extension_{bucket}_count"] += 1
        elif source == "http":
            if not row[4]:
                raise ValueError("Empty URL")
            sets["urls"].add(row[4])
            try:
                parsed = urlsplit(row[4])
                domain = parsed.hostname if parsed.scheme.lower() in {"http", "https"} else None
            except ValueError:
                domain = None
            if domain:
                sets["domains"].add(domain.lower())
            else:
                c["http_invalid_url_count"] += 1
        elif source == "email":
            if not row[4] or not row[7]:
                raise ValueError("Missing email sender/recipient")
            if any(not value.isascii() or not value.isdigit() for value in row[8:10]):
                raise ValueError("Email size/attachment count must be non-negative integers")
            size, attachments = map(int, row[8:10])
            c["email_size_bytes_sum"] += size
            c["email_size_bytes_max"] = max(c["email_size_bytes_max"], size)
            c["email_attachment_count"] += attachments
            c["email_with_attachment_count"] += int(attachments > 0)
            recipients = {address.strip().lower() for field in row[4:7]
                          for address in field.split(';') if address.strip()}
            if any('@' not in address or not all(address.rsplit('@', 1)) for address in recipients):
                raise ValueError("Invalid recipient address")
            sets["recipients"].update(recipients)
            internal = sum(a.rsplit('@', 1)[1] == 'dtaa.com' for a in recipients)
            external = len(recipients) - internal
            c["email_internal_recipient_count"] += internal
            c["email_external_recipient_count"] += external
            c["email_external_message_count"] += int(external > 0)

    def features(self, day):
        c, sets = self.counts, self.sets
        result = {name: c[name] for name in FEATURES}
        result.update(
            active_sources=sum(c[f"{s}_events"] > 0 for s in SOURCES),
            is_weekend=int(date.fromisoformat(day).weekday() >= 5),
            distinct_pcs=len(sets["pcs"]), first_second=self.first, last_second=self.last,
            activity_span_seconds=self.last - self.first, gap_count=self.gaps,
            gap_mean_seconds=self.gap_mean,
            gap_std_seconds=(max(self.gap_m2 / self.gaps, 0) ** .5 if self.gaps else 0),
            gap_max_seconds=self.gap_max,
            device_unmatched_connects=c["device_unmatched_connects"] + len(self.connections),
            file_distinct_names=len(sets["files"]), http_distinct_domains=len(sets["domains"]),
            http_distinct_urls=len(sets["urls"]), email_distinct_recipients=len(sets["recipients"]),
            email_size_bytes_mean=c["email_size_bytes_sum"] / max(c["email_events"], 1))
        for source in SOURCES:
            result[f"{source}_distinct_pcs"] = len(sets[f"{source}_pcs"])
            result[f"{source}_fraction"] = c[f"{source}_events"] / c["total_events"]
        return result

    def context(self):
        return {k: sorted(self.sets[k]) for k in ("pcs", "domains", "recipients")}
