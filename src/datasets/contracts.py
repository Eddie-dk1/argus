from __future__ import annotations

import re
from dataclasses import dataclass


NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_column_name(value: str) -> str:
    normalized = NORMALIZE_PATTERN.sub("_", str(value).strip().lower())
    return normalized.strip("_")


@dataclass(frozen=True)
class ColumnContract:
    canonical_name: str
    aliases: tuple[str, ...] = ()
    required: bool = True

    @property
    def accepted_names(self) -> tuple[str, ...]:
        names = (self.canonical_name, *self.aliases)
        return tuple(dict.fromkeys(normalize_column_name(name) for name in names))


@dataclass(frozen=True)
class SourceContract:
    name: str
    patterns: tuple[str, ...]
    columns: tuple[ColumnContract, ...]
    minimum_files: int = 1


@dataclass(frozen=True)
class ScenarioContract:
    name: str
    schema_status: str
    sources: tuple[SourceContract, ...]
    notes: str


def column(
    canonical_name: str,
    *aliases: str,
    required: bool = True,
) -> ColumnContract:
    return ColumnContract(
        canonical_name=canonical_name,
        aliases=tuple(aliases),
        required=required,
    )


SYNTHETIC_COLUMNS = (
    column("timestamp"),
    column("user_id"),
    column("source_ip"),
    column("destination_ip"),
    column("event_type"),
    column("login_status"),
    column("bytes_sent"),
    column("bytes_received"),
    column("protocol"),
    column("label"),
    column("anomaly_type"),
)

CERT_COMMON_COLUMNS = (
    column("id"),
    column("date", "timestamp"),
    column("user", "user_id"),
    column("pc", "computer"),
)

NETWORK_FLOW_COLUMNS = (
    column("flow_duration"),
    column("total_fwd_packets", "tot_fwd_pkts"),
    column("total_backward_packets", "tot_bwd_pkts", "total_bwd_packets"),
    column("flow_bytes_per_second", "flow_bytes_s", "flow_byts_s"),
    column("flow_packets_per_second", "flow_packets_s", "flow_pkts_s"),
    column("label"),
)


SCENARIO_CONTRACTS: dict[str, ScenarioContract] = {
    "synthetic": ScenarioContract(
        name="synthetic",
        schema_status="stable",
        sources=(
            SourceContract(
                name="synthetic_logs",
                patterns=("synthetic_logs.csv", "*/synthetic_logs.csv"),
                columns=SYNTHETIC_COLUMNS,
            ),
        ),
        notes="Schema is controlled by Argus and validated exactly.",
    ),
    "cert_r4_2": ScenarioContract(
        name="cert_r4_2",
        schema_status="publisher-r4.2-core-validated",
        sources=(
            SourceContract(
                name="logon",
                patterns=("logon.csv", "*/logon.csv"),
                columns=(*CERT_COMMON_COLUMNS, column("activity")),
            ),
            SourceContract(
                name="device",
                patterns=("device.csv", "*/device.csv"),
                columns=(*CERT_COMMON_COLUMNS, column("activity")),
            ),
            SourceContract(
                name="file",
                patterns=("file.csv", "*/file.csv"),
                columns=(
                    *CERT_COMMON_COLUMNS,
                    column("filename", "file"),
                    column("activity", required=False),
                ),
            ),
            SourceContract(
                name="http",
                patterns=("http.csv", "*/http.csv"),
                columns=(*CERT_COMMON_COLUMNS, column("url")),
            ),
            SourceContract(
                name="email",
                patterns=("email.csv", "*/email.csv"),
                columns=(
                    *CERT_COMMON_COLUMNS,
                    column("to"),
                    column("from"),
                    column("size"),
                    column("attachments", "attachment_count", required=False),
                ),
            ),
        ),
        notes=(
            "Core activity groups reconciled with publisher r4.2 README and headers. "
            "file.csv is removable-media copy activity without an activity field; "
            "email headers use attachments whereas README says attachment_count. "
            "Core schema validation is not full data or label-policy acceptance; "
            "LDAP/psychometric sources and answer observables are audited separately."
        ),
    ),
    "cicids2017": ScenarioContract(
        name="cicids2017",
        schema_status="stable-core-aliases",
        sources=(
            SourceContract(
                name="flows",
                patterns=("*.csv", "*/*.csv"),
                columns=NETWORK_FLOW_COLUMNS,
            ),
        ),
        notes=(
            "Core flow aliases accept original CICIDS2017 and abbreviated "
            "CICFlowMeter-V3 headings; the full feature manifest is data-driven."
        ),
    ),
    "cse_cic_ids2018": ScenarioContract(
        name="cse_cic_ids2018",
        schema_status="stable-core-aliases",
        sources=(
            SourceContract(
                name="flows",
                patterns=("*.csv", "*/*.csv"),
                columns=NETWORK_FLOW_COLUMNS,
            ),
        ),
        notes=(
            "The required core is shared with CICIDS2017. Daily extensions are "
            "accepted only after per-file schema inspection."
        ),
    ),
}
