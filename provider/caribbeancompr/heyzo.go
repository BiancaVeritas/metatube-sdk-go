package caribbeancompr

import (
	"regexp"
	"strings"
)

// heyzoNumber matches a HEYZO number in any spelling a filename or a caller may
// carry. The digits are the whole identity of the work.
var heyzoNumber = regexp.MustCompile(`(?i)^HEYZO[-_ ]?(\d{1,5})$`)

// heyzoDigits returns the zero-padded digits segment of a HEYZO number. Table
// keys are padded to four digits, the same shape the site's own numbers use.
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

// heyzoPageID resolves a HEYZO number to the page that carries the work.
func heyzoPageID(number string) (string, bool) {
	entry, ok := heyzoLookup(number)
	if !ok || entry.ID == "" {
		return "", false
	}
	return entry.ID, true
}

// heyzoNumberForPage is the reverse lookup: a page addressed by its own id still
// reports the number the work is known by, and picks up the verified title when
// the site's title is wrong (it occasionally mislabels a sequel volume).
func heyzoNumberForPage(pageID string) (number string, overrideTitle string, ok bool) {
	_, ids := heyzoLoad()
	digits, found := ids[pageID]
	if !found {
		return "", "", false
	}
	entry, found := heyzoLookup("HEYZO-" + digits)
	if !found {
		return "", "", false
	}
	return "HEYZO-" + digits, entry.OverrideTitle, true
}
