"""Run a DANCE workflow command without importing unrelated dependencies."""

import argparse
import runpy
import sys

COMMANDS = {
    'prepare-data': ('dance.data_preparation.prepare_cross_language_data', 'Prepare cross-language distractor data.'),
    'discover-neurons': ('dance.discovery.discover_bias_neurons', 'Discover bias-knowledge neurons.'),
    'analyze-neurons': ('dance.discovery.analyze_bias_neurons', 'Summarize discovered neurons.'),
    'bias-importance': ('dance.profiling.measure_bias_importance', 'Measure bias importance.'),
    'general-importance': ('dance.profiling.measure_general_importance', 'Measure general-knowledge sensitivity.'),
    'extract-activations': ('dance.profiling.extract_cross_language_activations', 'Extract low-resource neuron activations.'),
    'cross-language-importance': ('dance.profiling.measure_cross_language_importance', 'Measure cross-language activation-replacement effects.'),
    'neuron-contributions': ('dance.graph.measure_neuron_contributions', 'Generate contribution profiles for graph construction.'),
    'identify-modules': ('dance.graph.identify_neuron_modules', 'Find communities and candidate neuron pairs.'),
    'analyze-interactions': ('dance.graph.analyze_neuron_interactions', 'Evaluate pairwise neuron interactions.'),
    'interaction-stability': ('dance.graph.analyze_interaction_stability', 'Compare interactions across enhancement strengths.'),
    'optimize-coalitions': ('dance.graph.optimize_neuron_coalitions', 'Compute adaptive coalition weights.'),
    'evaluate-bias': ('dance.evaluation.evaluate_bias_benchmarks', 'Run the original bias/capability benchmark entry point.'),
    'evaluate-multilingual': ('dance.evaluation.evaluate_multilingual_benchmarks', 'Run multilingual benchmark evaluation.'),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="dance",
        description="Run a DANCE workflow stage. Pass --help after a command for its options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join(
            f"  {name:28} {description}" for name, (_, description) in COMMANDS.items()
        ),
    )
    if not argv:
        parser.print_help()
        return 0
    if argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
        return 0
    command, *arguments = argv
    if command not in COMMANDS:
        parser.error(f"unknown command: {command!r}; use --help to list commands")
    module, _ = COMMANDS[command]
    previous_argv = sys.argv
    try:
        sys.argv = [module, *arguments]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
