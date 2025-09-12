#!/bin/bash

# Define the server address
SERVER_URL="localhost:8001/quip"

# JSON payload to send
read -r -d '' PAYLOAD << EOM
{
  "format_quotes": true,
  "documents": [
    "A Bloom filter is a space-efficient probabilistic data structure, conceived by Burton Howard Bloom in 1970, that is used to test whether an element is a member of a set. False positive matches are possible, but false negatives are not",
    ".tes eht ni stnemele fo rebmun ro ezis eht fo tnednepedni ,ytilibaborp evitisop eslaf %1 a rof deriuqer era tnemele rep stib 01 naht rewef ,yllareneg eroM",
    "Testcase for data portraits as follows: A Bloom filter is a space-efficient probabilistic data structure. END PORTRAITS TEST CASE",
    "The ninth floor of the Miami-Dade pretrial detention facility is dubbed the \"forgotten floor.\""
  ]
}
EOM

# Make the POST request
curl -X POST "$SERVER_URL" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" | \
  jq 'map(del(.segments, .is_member, .chains, .chain_idxs, .quote_latex))'