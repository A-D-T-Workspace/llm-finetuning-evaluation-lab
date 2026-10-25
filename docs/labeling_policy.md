# Support Message Labeling Policy

## 1. Purpose

Assign each support message one primary route and an
independent escalation decision.

## 2. Routing Categories

- `account_support`: Routine login, password, account
  settings, and account email changes.
- `billing_support`: Charges, refunds, declined payments,
  receipts, and payment disputes.
- `account_security`: Unauthorized account access,
  compromised credentials, and account takeover.
- `order_support`: Order status, delivery, tracking,
  and arrival problems.
- `general_support`: General product questions and requests
  that do not fit another category.

## 3. Escalation Rules

Set escalation to `true` when the message indicates
suspected fraud, stolen payment details, unauthorized
transactions, phishing, or unauthorized account access.

Routine account, billing, delivery, and general questions
should not be escalated unless there is evidence of a
security incident or another defined risk.

## 4. Separate Decisions

The primary route describes which team should handle the
issue. Escalation describes whether the issue requires
urgent or specialist attention.

A message may therefore have a billing route and still
require escalation.

## 5. Ambiguous Cases

When a message involves multiple categories, choose the
route most directly responsible for resolving the user's
main issue. Apply escalation independently.

Review ambiguous examples and record the reason for the
chosen label before using them for training or evaluation.
