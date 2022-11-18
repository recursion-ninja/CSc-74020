## JSON Extractor

### A tool for parsing and joining 5e.tools monster stat block JSON file with the dndcombat.com elo JSON files


#### Building `curate-json`

  1.  Install the [Haskell] build tool [`cabal` ][cabal]

  2.  Build the project by invoking the following command:

      `cabal install`


#### Using `curate-json`

The JSON Curator expects a list of *"ranking" files* and a list of *"stat-block" files*.

An example invocation of `curate-json`:

```
$ curate-json -r ranks-1.json -r ranks-2.json -s stats.json
```

This invocation will parse data from the JSON files, match the ranking so the stats by creature name, and write the results to a 'dnd-5e-monsters.csv' file in the working directory.
Optionally, a different file path can be supplied with either the '-o' or '--output` flags.

[haskel]: https://www.haskell.org/
[cabal ]: https://www.haskell.org/cabal/download.html