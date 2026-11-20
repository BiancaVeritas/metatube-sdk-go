package caribbeancompr

import (
	_ "embed"
	"encoding/json"
	"sync"
)

// heyzoIndexData maps a HEYZO number to the CaribbeancomPR page that carries the
// same work, because the site itself never prints HEYZO numbers — it only has
// its own page ids. The table is generated out of tree (见 amane-work/heyzo_map).
//
//go:embed heyzo_index.json
var heyzoIndexData []byte

// heyzoEntry is one row: the page id plus the title, actor and any verified
// title recorded during the crawl.
type heyzoEntry struct {
	ID            string `json:"id"`
	Title         string `json:"title"`
	Actor         string `json:"actor"`
	OverrideTitle string `json:"override_title"`
}

var (
	heyzoOnce  sync.Once
	heyzoTable map[string]heyzoEntry
	heyzoByID  map[string]string
)

// heyzoLoad parses the embedded table once and builds the reverse page-id index.
// A malformed table degrades to empty maps so the provider keeps working.
func heyzoLoad() (map[string]heyzoEntry, map[string]string) {
	heyzoOnce.Do(func() {
		var raw map[string]heyzoEntry
		entries := make(map[string]heyzoEntry)
		if err := json.Unmarshal(heyzoIndexData, &raw); err == nil {
			entries = raw
		}
		reversed := make(map[string]string, len(entries))
		for digits, entry := range entries {
			if entry.ID != "" {
				reversed[entry.ID] = digits
			}
		}
		heyzoTable, heyzoByID = entries, reversed
	})
	return heyzoTable, heyzoByID
}

// heyzoLookup resolves a number to its table row.
func heyzoLookup(number string) (heyzoEntry, bool) {
	digits, ok := heyzoDigits(number)
	if !ok {
		return heyzoEntry{}, false
	}
	table, _ := heyzoLoad()
	entry, found := table[digits]
	return entry, found
}
