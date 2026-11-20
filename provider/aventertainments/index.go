package aventertainments

import (
	_ "embed"
	"encoding/json"
	"strings"
	"sync"
)

// proIndexData is the offline number → `pro` mapping, generated out of tree
// from the site's own detail pages (见 amane-work/ave_index/crawl_ave.py).
//
//go:embed pro_index.json
var proIndexData []byte

var (
	proIndexOnce  sync.Once
	proIndexTable map[string]int
)

// proIndex lazily parses the embedded table. A malformed table degrades to an
// empty index so the provider keeps working through the site's search page.
func proIndex() map[string]int {
	proIndexOnce.Do(func() {
		var raw map[string]int
		index := make(map[string]int)
		if err := json.Unmarshal(proIndexData, &raw); err == nil {
			index = raw
		}
		proIndexTable = index
	})
	return proIndexTable
}

// lookupPro resolves a number to the site's `pro` id.
//
// The lookup is exact and only case/space insensitive, never separator
// insensitive: the site treats `AND-001` and `AND001`, or `F-B001` and `FB001`,
// as different products and the table holds both spellings. Folding them would
// hand one product's id to the other's number, so a miss is left to the search
// page instead.
func lookupPro(number string) (int, bool) {
	number = strings.TrimSpace(number)
	if number == "" {
		return 0, false
	}
	pro, ok := proIndex()[strings.ToUpper(number)]
	return pro, ok
}

// proCount reports the table size, for tests and startup diagnostics.
func proCount() int { return len(proIndex()) }
