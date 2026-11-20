package d2pass

import (
	_ "embed"
	"encoding/json"
	"strconv"
	"strings"
	"sync"
)

// heyzoIndexData maps a HEYZO number to the D2PASS product carrying it. The
// site's search answers titles and actors only, so this table is the only way a
// number reaches a product at all. It is generated out of tree (见
// amane-work/heyzo_map).
//
//go:embed heyzo_index.json
var heyzoIndexData []byte

var (
	heyzoOnce  sync.Once
	heyzoTable map[string]string
)

// heyzoTable load parses the embedded table once, as number → product id.
// A malformed table degrades to an empty map; the provider then simply reports
// every number as not found.
func heyzoIndex() map[string]string {
	heyzoOnce.Do(func() {
		var raw map[string]struct {
			ID int `json:"id"`
		}
		table := make(map[string]string)
		if err := json.Unmarshal(heyzoIndexData, &raw); err == nil {
			table = make(map[string]string, len(raw))
			for digits, entry := range raw {
				if entry.ID > 0 {
					table[digits] = strconv.Itoa(entry.ID)
				}
			}
		}
		heyzoTable = table
	})
	return heyzoTable
}

// heyzoDigits returns the zero-padded digits of a HEYZO number, matching the
// table's key shape.
func heyzoDigits(number string) (string, bool) {
	ss := heyzoNumber.FindStringSubmatch(strings.TrimSpace(number))
	if len(ss) != 2 {
		return "", false
	}
	digits := strings.TrimLeft(ss[1], "0")
	if digits == "" {
		return "", false
	}
	if len(digits) < 4 {
		digits = strings.Repeat("0", 4-len(digits)) + digits
	}
	return digits, true
}

// heyzoProductID resolves a HEYZO number to its product id.
func heyzoProductID(number string) (string, bool) {
	digits, ok := heyzoDigits(number)
	if !ok {
		return "", false
	}
	pid, found := heyzoIndex()[digits]
	return pid, found
}

// productID resolves whatever the caller passed — a product id or a HEYZO
// number — to a product id.
func productID(value string) (string, bool) {
	if pid, ok := heyzoProductID(value); ok {
		return pid, true
	}
	value = strings.TrimSpace(value)
	if value == "" || !isDigits(value) {
		return "", false
	}
	return value, true
}

func isDigits(value string) bool {
	for _, r := range value {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
