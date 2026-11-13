import argparse
import json
from pathlib import Path

from src.reporting.adapters import HASHES, validate_source
from src.reporting.pipeline import reporting_plan, generate_report, verify_report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plan, validate, generate or verify scientific reports without executing experiments.')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--plan', action='store_true')
    action.add_argument('--validate', type=Path)
    action.add_argument('--generate', type=Path)
    action.add_argument('--verify-report', type=Path)
    parser.add_argument('--experiment', choices=tuple(HASHES))
    parser.add_argument('--mode', choices=('final', 'diagnostic'), default='final')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--ledger', type=Path, help='Experiment 1 archived controller ledger, inside source root')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--verify-inputs', action='store_true')
    args = parser.parse_args(argv)
    if (args.validate or args.generate) and not args.experiment:
        parser.error('--experiment is required for source validation/report generation.')
    if args.generate and not args.output:
        parser.error('--generate requires a fresh --output directory.')
    if not args.generate and (args.output or args.approval):
        parser.error('--output/--approval require --generate.')
    if args.verify_inputs and not args.verify_report:
        parser.error('--verify-inputs requires --verify-report.')
    if args.verify_report:
        result = verify_report(args.verify_report, verify_inputs=args.verify_inputs)
    elif args.validate:
        report = validate_source(args.experiment, args.validate, ledger_path=args.ledger)
        result = {k: report[k] for k in ('experiment', 'protocol_hash', 'source_inventory_hash', 'complete',
            'issues', 'planned_cells', 'observed_cells', 'planned_records', 'complete_records', 'artificial_fixture')}
    elif args.generate:
        result = generate_report(args.experiment, args.generate, args.output, mode=args.mode,
                                 approval=args.approval, ledger_path=args.ledger, dpi=args.dpi)
    else:
        result = reporting_plan(args.experiment)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
