#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_DIR="${1:-"$ROOT_DIR/../mining-massive-datasets-history-fixture"}"

if [[ "$FIXTURE_DIR" != *fixture* ]]; then
  echo "Refusing to recreate '$FIXTURE_DIR': fixture path must contain 'fixture'." >&2
  exit 1
fi

case "$FIXTURE_DIR" in
  "/"|"$HOME"|"$ROOT_DIR"|"$ROOT_DIR/")
    echo "Refusing to recreate unsafe fixture path: $FIXTURE_DIR" >&2
    exit 1
    ;;
esac

rm -rf "$FIXTURE_DIR"
mkdir -p "$FIXTURE_DIR"
cd "$FIXTURE_DIR"

git init -q
git config user.name "History Fixture"
git config user.email "history.fixture@example.invalid"

commit_fixture() {
  local author_name="$1"
  local author_email="$2"
  local date="$3"
  local message="$4"
  local file_path="$5"
  local body="$6"

  mkdir -p "$(dirname "$file_path")"
  printf '%s\n' "$body" > "$file_path"
  git add "$file_path"
  GIT_AUTHOR_NAME="$author_name" \
    GIT_AUTHOR_EMAIL="$author_email" \
    GIT_AUTHOR_DATE="$date" \
    GIT_COMMITTER_NAME="$author_name" \
    GIT_COMMITTER_EMAIL="$author_email" \
    GIT_COMMITTER_DATE="$date" \
    git commit -q -m "$message"
}

commit_fixture \
  "History Fixture" \
  "history.fixture@example.invalid" \
  "2026-04-04T09:30:00+07:00" \
  "fixture: initialize synthetic week 9-12 history" \
  "README.md" \
  "# Synthetic commit history fixture

This repository is generated for testing commit-history tooling only.
It intentionally contains backdated fixture commits and should not be used as real project history."

commit_fixture \
  "Dinh Manh Hung Fixture" \
  "hung.fixture@example.invalid" \
  "2026-04-04T10:00:00+07:00" \
  "week 9: add combined telegram data integration and source metadata" \
  "hung/week09.md" \
  "Week 9 Hung fixture commit.

Plan row: Ho tro fix data / output; chuan bi demo so bo."

commit_fixture \
  "Bao Fixture" \
  "bao.fixture@example.invalid" \
  "2026-04-06T10:00:00+07:00" \
  "week 9: add similarity histogram and cluster graph visuals" \
  "bao/week09.md" \
  "Week 9 Bao fixture commit.

Plan rows: Ve histogram similarity; ve graph clusters."

commit_fixture \
  "Dinh Manh Hung Fixture" \
  "hung.fixture@example.invalid" \
  "2026-04-11T10:00:00+07:00" \
  "week 10: add scalability benchmark and full combined pipeline runner" \
  "hung/week10.md" \
  "Week 10 Hung fixture commit.

Plan rows: Test scalability tren du lieu lon hon; so sanh runtime LSH vs brute-force; toi uu code / memory."

commit_fixture \
  "Bao Fixture" \
  "bao.fixture@example.invalid" \
  "2026-04-12T10:00:00+07:00" \
  "week 10: add performance charts and runtime summary table" \
  "bao/week10.md" \
  "Week 10 Bao fixture commit.

Plan row: Ve bieu do hieu nang."

commit_fixture \
  "Dinh Manh Hung Fixture" \
  "hung.fixture@example.invalid" \
  "2026-04-18T10:00:00+07:00" \
  "week 11: add streamlit dashboard and demo cases" \
  "hung/week11.md" \
  "Week 11 Hung fixture commit.

Plan rows: Chuan bi demo cases; test demo."

commit_fixture \
  "Bao Fixture" \
  "bao.fixture@example.invalid" \
  "2026-04-20T10:00:00+07:00" \
  "week 11: add slide outline and consolidated charts tables" \
  "bao/week11.md" \
  "Week 11 Bao fixture commit.

Plan rows: Lam slide; tong hop bieu do / bang."

commit_fixture \
  "Dinh Manh Hung Fixture" \
  "hung.fixture@example.invalid" \
  "2026-04-25T10:00:00+07:00" \
  "week 12: add final report and end-to-end project documentation" \
  "hung/week12.md" \
  "Week 12 Hung fixture commit.

Plan rows: Viet phan thuat toan; ghep bao cao final."

commit_fixture \
  "Bao Fixture" \
  "bao.fixture@example.invalid" \
  "2026-04-26T10:00:00+07:00" \
  "week 12: add dataset results report and demo practice checklist" \
  "bao/week12.md" \
  "Week 12 Bao fixture commit.

Plan rows: Viet phan dataset + ket qua; practice demo."

git log --date=iso-strict --pretty=format:'%h%x09%an%x09%ae%x09%ad%x09%s'
