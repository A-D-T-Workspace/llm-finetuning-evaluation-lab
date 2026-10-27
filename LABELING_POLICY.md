# Labeling Policy — Customer Support Routing

## 1. Purpose

Define consistent ground-truth labels for evaluating customer-support routing and escalation predictions.

## 2. Route Labels

- `account_support`: Account access, profile changes, or routine account assistance.
- `billing_support`: Payments, invoices, charges, refunds, or billing disputes.
- `account_security`: Unauthorized access, compromised credentials, suspected account takeover, or account-security threats.
- `order_support`: Order status, delivery, shipping, or order-related problems.
- `general_support`: General questions that do not clearly belong to another route.

## 3. Escalation Labels

Set `escalation` to `true` when the case requires urgent human review or intervention because of a security threat, suspected fraud, unauthorized activity, or another explicitly defined high-risk condition.

Set it to `false` for routine requests that can follow the normal support workflow.

Escalation is a separate decision from route selection. A case must not be escalated merely because it belongs to a particular route.

Escalation decisions must be based on the risk and urgency described in the case, not solely on its route. Credible unauthorized activity or a security threat may require escalation even when the full extent of harm is unknown. Routine security questions and unsubstantiated concerns should not automatically be escalated.

## 4. Ambiguous Cases

- Assign the route based on the customer's primary issue.
- Use the most specific applicable route when evidence supports it.
- Do not infer facts that the customer did not provide.
- If the available information is insufficient, follow a documented annotation rule rather than guessing.
- Record disagreements and resolve them before using affected examples as evaluation ground truth.

## 5. Evaluation Integrity

- Keep the original 20-case dataset unchanged as version 1.
- Version any expanded dataset separately.
- Preserve fixed train, validation, and test partitions when comparing experiments.
- Do not modify labels or prompts merely to improve held-out test scores.
- Report dataset size and class distribution alongside evaluation metrics.
- Treat results from very small evaluation sets as preliminary.

## 6. Quality Review

Before adding an example, verify that:
1. The route matches the primary customer issue.
2. The escalation label follows the policy.
3. The example is not an accidental duplicate.
4. The wording does not reveal the expected answer through artificial formatting.
5. The labels can be justified from the case text alone.
