# @yaml
# tdd: true
# signature: tdd_repro ["<target_file>"] ["<target_function_name>"] [<decl_lineno>]
# docstring: Reproduces the bug by running bug-specific tests. Provide optional target arguments to get runtime context for the target function.
# arguments:
#  target_file:
#     type: string
#     description: The file containing the target function, relative to CWD. If provided, target_function_name is also required.
#     required: false
#  target_function_name:
#     type: string
#     description: The UNQUALIFIED(!) name of the target function or method.
#     required: false
#  decl_lineno:
#     type: integer
#     description: The lineno of target_function's declaration. Only required if target_function_name is ambiguous within the file.
#     required: false
tdd_repro() {
    # set -euo pipefail
    if [ -z "$TEST_CMD_FAIL_TO_PASS" ]; then
        echo "ERROR: env var \$TEST_CMD_FAIL_TO_PASS missing."
        exit 1
    fi
    pushd $REPO_ROOT > /dev/null
    echo -e "Running tests to reproduce the bug (from $PWD):\n >$TEST_CMD_FAIL_TO_PASS\n"
    if [ $# -ge 1 ]; then
        line_no=${3:-0}
        export TDD_TRACE_TARGET_CONFIG="{ \"target_file\": \"$1\", \"target_function_name\": \"$2\", \"decl_lineno\": $line_no}"
    fi
    eval "$TEST_CMD_FAIL_TO_PASS"
    popd > /dev/null
}

# # @yaml
# # docstring: Run all tests to check for regressions. This might execute multiple individual test runs.
# # tdd: true
# tdd_run_all() {
#     # Assert that the file exists
#     if [ ! -f "$PASS_TO_PASS_FILE" ]; then
#         echo "ERROR: File $PASS_TO_PASS_FILE not found."
#         exit 1
#     fi

#     echo "tdd_run_all is disabled for now. Don't use it."
#     exit 1

#     # Read PASS_TO_PASS_FILE line by line and execute the command
#     while IFS= read -r line || [[ -n "$line" ]]; do
#         # Trim leading and trailing whitespace
#         line=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        
#         # Skip empty lines
#         if [ -n "$line" ]; then
#             eval "$TEST_CMD $line"
#         fi
#     done < "$PASS_TO_PASS_FILE"
# }
