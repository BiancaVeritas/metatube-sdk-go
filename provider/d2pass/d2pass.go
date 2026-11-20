// Package d2pass serves the D2PASS catalog, which is the only source left for
// studios whose own site is gone — HEYZO among them.
//
// Each product carries a `product_url` pointing back at the studio's own page,
// and that URL is what makes a number lookup possible: the site's search only
// answers by title and actor. The offline number → product_id table therefore
// comes first, and the search page is not used at all for a number.
package d2pass

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/gocolly/colly/v2"
	"golang.org/x/text/language"

	"github.com/metatube-community/metatube-sdk-go/common/parser"
	"github.com/metatube-community/metatube-sdk-go/model"
	"github.com/metatube-community/metatube-sdk-go/provider"
	"github.com/metatube-community/metatube-sdk-go/provider/internal/scraper"
)

var (
	_ provider.MovieProvider = (*D2PASS)(nil)
	_ provider.MovieSearcher = (*D2PASS)(nil)
)

const (
	Name = "D2PASS"
	// Priority sits just below the sites that still have one: this catalog is a
	// fallback for works nobody else can supply, not a first choice.
	Priority = 1000 - 3
)

const (
	baseURL  = "https://www.d2pass.com/"
	movieURL = "https://www.d2pass.com/product/movies/%s"
	imageURL = "https://images.d2pass.com/images/meta/movies/flash/%s.webp"
)

// heyzoNumber is the only number this provider claims: every other studio in
// the catalog still has its own site, which serves it better.
var heyzoNumber = regexp.MustCompile(`(?i)^HEYZO[-_ ]?(\d{1,5})$`)

// studioNumber reads the number out of a product_url, which points at the
// studio page: `.../moviepages/0333/index.html` → HEYZO-0333.
var studioNumber = regexp.MustCompile(`(?i)/(?:moviepages|movie)/([0-9A-Za-z_-]+)`)

type D2PASS struct {
	*scraper.Scraper
}

func New() *D2PASS {
	return &D2PASS{scraper.NewDefaultScraper(Name, baseURL, Priority, language.Japanese)}
}

// NormalizeMovieID accepts a product id, or a HEYZO number the table knows.
func (d2 *D2PASS) NormalizeMovieID(id string) string {
	id = strings.TrimSpace(id)
	if pid, ok := productID(id); ok {
		return pid
	}
	if regexp.MustCompile(`^\d+$`).MatchString(id) {
		return id
	}
	return ""
}

// NormalizeMovieKeyword keeps the provider to HEYZO numbers; anything else
// returns empty so the engine skips it.
func (d2 *D2PASS) NormalizeMovieKeyword(keyword string) string {
	keyword = strings.TrimSpace(keyword)
	if !heyzoNumber.MatchString(keyword) {
		return ""
	}
	return strings.ToUpper(keyword)
}

func (d2 *D2PASS) GetMovieInfoByID(id string) (*model.MovieInfo, error) {
	if pid, ok := productID(id); ok {
		return d2.getByProductID(pid)
	}
	return d2.getByProductID(id)
}

func (d2 *D2PASS) ParseMovieIDFromURL(rawURL string) (string, error) {
	ss := regexp.MustCompile(`/product/movies/(\d+)`).FindStringSubmatch(rawURL)
	if len(ss) != 2 {
		return "", fmt.Errorf("d2pass: not a product url: %s", rawURL)
	}
	return ss[1], nil
}

func (d2 *D2PASS) GetMovieInfoByURL(rawURL string) (*model.MovieInfo, error) {
	id, err := d2.ParseMovieIDFromURL(rawURL)
	if err != nil {
		return nil, err
	}
	return d2.getByProductID(id)
}

// SearchMovie answers a HEYZO number through the table.
//
// A miss reports not found rather than falling back to the search page: that
// page matches titles and actor names, and a number given to it returns either
// nothing or an unrelated work.
func (d2 *D2PASS) SearchMovie(keyword string) ([]*model.MovieSearchResult, error) {
	pid, ok := productID(keyword)
	if !ok {
		return nil, provider.ErrInfoNotFound
	}
	info, err := d2.getByProductID(pid)
	if err != nil || info == nil || !info.IsValid() {
		return nil, provider.ErrInfoNotFound
	}
	return []*model.MovieSearchResult{info.ToSearchResult()}, nil
}

// getByProductID scrapes one product page.
func (d2 *D2PASS) getByProductID(pid string) (info *model.MovieInfo, err error) {
	info = &model.MovieInfo{
		ID:            pid,
		Provider:      Name,
		Homepage:      fmt.Sprintf(movieURL, pid),
		Actors:        []string{},
		PreviewImages: []string{},
		Genres:        []string{},
	}

	c := d2.ClonedCollector()
	c.OnXML(`//script`, func(e *colly.XMLElement) {
		if info.Title == "" {
			fillFromPayload(info, e.Text)
		}
	})
	if err = c.Visit(info.Homepage); err != nil {
		return nil, err
	}
	if info.Title == "" {
		return nil, provider.ErrInfoNotFound
	}
	return info, nil
}

// fillFromPayload reads the product object out of the JSON the page streams.
// The fragment is escaped, so it is unescaped once and then decoded as a whole
// instead of being picked apart field by field.
func fillFromPayload(info *model.MovieInfo, text string) {
	product := productObject(text)
	if product == nil {
		return
	}
	info.Title = field(product, "title")
	info.Summary = field(product, "description")
	info.Maker = field(product, "site_name")
	if pid := numberField(product, "product_id"); pid != "" {
		info.ID = pid
		poster := fmt.Sprintf(imageURL, pid)
		info.CoverURL, info.ThumbURL = poster, poster
	}
	if actor := field(product, "actor_name"); actor != "" {
		info.Actors = []string{actor}
	}
	if tags := field(product, "uc_list"); tags != "" {
		info.Genres = strings.Split(tags, ":")
	}
	info.ReleaseDate = parser.ParseDate(field(product, "release_date"))
	if seconds, ok := product["duration"].(float64); ok && seconds > 0 {
		info.Runtime = int(seconds) / 60
	}
	if number := studioNumberFromURL(field(product, "product_url")); number != "" {
		info.Number = number
	}
}

// productMarker is where the product object begins in the streamed payload.
const productMarker = `"product":{"product_id":`

func productObject(text string) map[string]any {
	unescaped := strings.ReplaceAll(text, `\"`, `"`)
	index := strings.Index(unescaped, productMarker)
	if index < 0 {
		return nil
	}
	var product map[string]any
	start := index + len(`"product":`)
	if err := json.NewDecoder(strings.NewReader(unescaped[start:])).Decode(&product); err != nil {
		return nil
	}
	return product
}

func field(product map[string]any, key string) string {
	value, _ := product[key].(string)
	return strings.TrimSpace(value)
}

func numberField(product map[string]any, key string) string {
	value, ok := product[key].(float64)
	if !ok {
		return ""
	}
	return fmt.Sprintf("%d", int(value))
}

// studioNumberFromURL turns the studio page URL into the work's number.
func studioNumberFromURL(rawURL string) string {
	ss := studioNumber.FindStringSubmatch(rawURL)
	if len(ss) != 2 {
		return ""
	}
	return "HEYZO-" + ss[1]
}

func init() {
	provider.Register(Name, New)
}
