import requests,json,time,re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path

class CNUNoticeCrawler:
    def __init__(self):
        self.base_url = "https://eng.cnu.ac.kr"
        self.notice_url = "https://eng.cnu.ac.kr/eng/information/notice.do"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def get_notice_list(self, page=1):
        """공지사항 목록을 가져오는 함수"""
        limit = 10  # 한 페이지당 게시글 수 (기본값이 10임)
        offset = (page - 1) * limit

        params = {
            'mode': 'list',
            'articleLimit': limit,
            'article.offset': offset
        }

        try:
            print(f"  페이지 {page} 요청 중... (offset={offset})")
            response = self.session.get(self.notice_url, params=params, timeout=30)
            print(f"  실제 요청된 URL: {response.url}")
            response.raise_for_status()
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.content, 'html.parser')
            print(f"  페이지 {page} HTML 파싱 완료")
            return soup

        except requests.RequestException as e:
            print(f"  페이지 {page} 요청 실패: {e}")
            return None
        except Exception as e:
            print(f"  페이지 {page} 처리 오류: {e}")
            return None

    def extract_notice_links(self, soup):
        """공지사항 목록에서 개별 공지사항 링크를 추출"""
        notices = []

        try:
            # 다양한 테이블 구조 시도
            table_selectors = [
                'table.boardList tbody tr',
                'table tbody tr',
                '.board-list tbody tr',
                '.list-table tbody tr',
                'tbody tr'
            ]

            rows = []
            for selector in table_selectors:
                rows = soup.select(selector)
                if rows:
                    print(f"    테이블 찾음: {selector} - {len(rows)}개 행")
                    break

            if not rows:
                print("    테이블을 찾을 수 없음")
                # 디버깅을 위해 페이지 내용 일부 출력
                print("    페이지 내용 샘플:")
                text_content = soup.get_text()[:500]
                print(f"    {text_content}")
                return notices

            for idx, row in enumerate(rows):
                try:
                    # 제목 링크 찾기 - 다양한 방법 시도
                    title_link = None

                    # 방법 1: a 태그 직접 찾기
                    links = row.find_all('a')
                    for link in links:
                        href = link.get('href', '')
                        if 'view.do' in href or 'articleNo' in href:
                            title_link = link
                            break

                    # 방법 2: td에서 a 태그 찾기
                    if not title_link:
                        tds = row.find_all('td')
                        for td in tds:
                            link = td.find('a')
                            if link and link.get('href'):
                                title_link = link
                                break

                    if not title_link:
                        continue

                    href = title_link.get('href')
                    if not href:
                        continue

                    # 상대 경로를 절대 경로로 변환
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        full_url = urljoin(self.notice_url, href)

                    # 제목 추출
                    title = title_link.get_text(strip=True)
                    if not title or len(title) < 2:
                        continue

                    # 등록일 추출
                    date = ""
                    tds = row.find_all('td')
                    for td in tds:
                        text = td.get_text(strip=True)
                        # 날짜 형식 패턴들 (YY.MM.DD, YYYY-MM-DD 등)
                        date_patterns = [
                            r'\d{2}\.\d{2}\.\d{2}',
                            r'\d{4}-\d{2}-\d{2}',
                            r'\d{4}\.\d{2}\.\d{2}',
                            r'\d{2}/\d{2}/\d{2}'
                        ]

                        for pattern in date_patterns:
                            match = re.search(pattern, text)
                            if match:
                                date = match.group()
                                break
                        if date:
                            break

                    notice_info = {
                        'title': title,
                        'url': full_url,
                        'date': date
                    }

                    notices.append(notice_info)
                    print(f"    공지사항 추출: {title[:30]}...")

                except Exception as e:
                    print(f"    행 {idx} 처리 오류: {e}")
                    continue

            print(f"    총 {len(notices)}개 공지사항 추출 완료")

        except Exception as e:
            print(f"  공지사항 링크 추출 전체 오류: {e}")

        return notices

    def clean_content(self, content):
        """내용에서 불필요한 부분 제거하고 정리"""
        if not content:
            return ""

        # 불필요한 텍스트 패턴들 제거
        unwanted_patterns = [
            r'메뉴.*?바로가기',
            r'주요.*?바로가기',
            r'Copyright.*?All rights reserved',
            r'이전글|다음글',
            r'목록으로|목록보기',
            r'첨부파일.*?다운로드',
            r'조회수.*?\d+',
            r'등록일.*?\d{2}\.\d{2}\.\d{2}',
            r'작성자.*?공과대학',
            r'HOME.*?공지사항',
            r'QUICK MENU.*',
            r'SITEMAP.*',
            r'개인정보처리방침.*',
            r'홈페이지.*?운영정책',
            r'SNS.*?바로가기',
            r'전체메뉴.*',
            r'검색.*?버튼',
            r'Language.*?Translation',
            r'번호\s*제목\s*첨부\s*작성자\s*등록일\s*조회수',
            r'페이지.*?이동',
            r'이전\s*다음\s*목록'
        ]

        # 패턴 제거
        cleaned = content
        for pattern in unwanted_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # 연속된 공백과 줄바꿈 정리
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # 빈 줄 정리
        cleaned = re.sub(r'\s+', ' ', cleaned)  # 연속된 공백 정리
        cleaned = re.sub(r'\n+', '\n', cleaned)  # 연속된 줄바꿈 정리

        # 앞뒤 공백 제거
        cleaned = cleaned.strip()

        # 너무 짧은 내용은 제외
        if len(cleaned) < 20:
            return ""

        return cleaned

    def is_title_duplicate(self, content, title):
        """내용이 제목과 중복되는지 확인"""
        if not content or not title:
            return False

        # 제목에서 특수문자 제거하고 비교
        title_clean = re.sub(r'[^\w\s]', '', title)
        content_clean = re.sub(r'[^\w\s]', '', content)

        # 내용의 첫 부분이 제목과 70% 이상 일치하면 중복으로 판단
        content_first = content_clean[:len(title_clean)]

        # 문자 단위로 일치도 계산
        matches = sum(1 for a, b in zip(title_clean.lower(), content_first.lower()) if a == b)
        similarity = matches / len(title_clean) if len(title_clean) > 0 else 0

        return similarity > 0.7

    def get_notice_content(self, notice_url, notice_title=""):
        """개별 공지사항의 내용을 가져오는 함수 (개선됨)"""
        try:
            print(f"      내용 가져오는 중...")
            response = self.session.get(notice_url, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.content, 'html.parser')

            # 불필요한 요소들을 먼저 제거
            unwanted_selectors = [
                'nav', 'header', 'footer', '.nav', '.header', '.footer',
                '.menu', '.gnb', '.lnb', '.snb', '.breadcrumb',
                '.btn', '.button', 'button',
                '.quick-menu', '.sitemap', '.copyright',
                '.social', '.sns', '.share',
                'script', 'style', 'noscript',
                '.prev-next', '.board-nav'  # 이전/다음 네비게이션 제거
            ]

            for selector in unwanted_selectors:
                for elem in soup.select(selector):
                    elem.decompose()

            # 디버깅: 페이지 구조 분석
            print(f"      페이지 구조 분석 중...")

            # 1단계: 더 구체적인 선택자들로 내용 찾기
            content_selectors = [
                # 게시판 상세보기 전용 선택자들
                '.board-view .view-content',
                '.board-detail .content',
                '.article-view .content',
                '.post-view .content',
                '.notice-view .content',

                # 테이블 기반 게시판 (더 구체적)
                'table.view-table tr:has(td[colspan]) td',
                'table.boardView tr:has(td[colspan]) td',
                'table.detailView tr:has(td[colspan]) td',

                # 일반적인 내용 영역
                '.view-content',
                '.article-content',
                '.post-content',
                '.content-area',
                '.board-content',

                # ID 기반 선택자
                '#content .view',
                '#articleContent',
                '#postContent'
            ]

            content = ""
            content_found = False

            for selector in content_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(separator='\n', strip=True)

                    # 충분한 내용이 있고, 제목과 중복되지 않는지 확인
                    if (len(text) > 50 and
                            not self.is_title_duplicate(text, notice_title) and
                            any(char in text for char in '.,!?')):
                        content = text
                        content_found = True
                        print(f"      내용 발견: {selector}")
                        break

                if content_found:
                    break

            # 2단계: 테이블에서 colspan이 있는 td 찾기 (더 정확한 방법)
            if not content_found:
                print(f"      테이블에서 colspan 기반 내용 검색...")

                # colspan 속성이 있는 td는 보통 내용을 담고 있음
                colspan_tds = soup.find_all('td', {'colspan': True})

                for td in colspan_tds:
                    text = td.get_text(separator='\n', strip=True)
                    if (len(text) > 100 and
                            not self.is_title_duplicate(text, notice_title) and
                            '공지사항' not in text[:50]):  # 헤더 부분 제외
                        content = text
                        content_found = True
                        print(f"      colspan 테이블에서 내용 발견")
                        break

            # 3단계: 모든 td에서 가장 긴 의미있는 내용 찾기
            if not content_found:
                print(f"      전체 테이블에서 내용 검색...")

                tds = soup.find_all('td')
                candidates = []

                for td in tds:
                    text = td.get_text(separator='\n', strip=True)

                    # 실제 내용으로 보이는 조건들
                    if (len(text) > 80 and
                            not self.is_title_duplicate(text, notice_title) and
                            not any(skip in text for skip in ['번호', '제목', '작성자', '조회수', '등록일']) and
                            any(keyword in text for keyword in
                                ['일시', '장소', '대상', '안내', '신청', '문의', '개최', '실시', '바랍니다', '하오니', '참가', '접수'])):
                        candidates.append((text, len(text)))

                # 가장 긴 후보를 선택
                if candidates:
                    content = max(candidates, key=lambda x: x[1])[0]
                    content_found = True
                    print(f"      테이블에서 {len(candidates)}개 후보 중 최적 내용 선택")

            # 4단계: 전체 페이지에서 패턴 매칭으로 내용 추출
            if not content_found:
                print(f"      전체 페이지 패턴 매칭...")
                body_text = soup.get_text(separator='\n', strip=True)
                lines = body_text.split('\n')

                content_lines = []
                start_collecting = False

                for line in lines:
                    line = line.strip()

                    # 내용 시작을 나타내는 키워드들
                    start_keywords = ['안내드립니다', '알려드립니다', '공지합니다', '개최합니다']

                    if any(keyword in line for keyword in start_keywords):
                        start_collecting = True

                    if start_collecting and len(line) > 10:
                        # 불필요한 라인 제외
                        if not any(skip in line for skip in ['메뉴', '바로가기', 'Copyright', '조회수', '이전글', '다음글']):
                            content_lines.append(line)

                    # 너무 많이 수집하면 중단
                    if len(content_lines) > 20:
                        break

                if content_lines:
                    content = '\n'.join(content_lines)
                    print(f"      패턴 매칭으로 {len(content_lines)}줄 추출")

            # 5단계: 최후의 수단 - 페이지에서 특정 키워드 포함 문단 추출
            if not content:
                print(f"      키워드 기반 최종 추출 시도...")

                # 모든 p 태그와 div 태그에서 검색
                for tag in soup.find_all(['p', 'div']):
                    text = tag.get_text(strip=True)
                    if (len(text) > 50 and
                            any(keyword in text for keyword in ['일시:', '장소:', '대상:', '문의:', '신청']) and
                            not self.is_title_duplicate(text, notice_title)):
                        content = text
                        print(f"      키워드 기반 내용 발견")
                        break

            # 내용 정리
            if content:
                content = self.clean_content(content)

                # 최종 중복 체크
                if self.is_title_duplicate(content, notice_title):
                    print(f"      ⚠️  추출된 내용이 제목과 중복됨, 내용 없음으로 처리")
                    content = ""

                # 내용이 너무 길면 자르기
                if len(content) > 1500:
                    content = content[:1500] + "..."

            if not content:
                print(f"      ❌ 내용을 찾을 수 없음")
            else:
                print(f"      ✅ 내용 추출 완료 ({len(content)}자)")

            return content

        except requests.RequestException as e:
            print(f"      내용 요청 실패: {e}")
            return ""
        except Exception as e:
            print(f"      내용 추출 오류: {e}")
            return ""

    def crawl_notices(self, max_pages=10):
        """공지사항을 크롤링하는 메인 함수"""
        all_notices = []

        print(f"=== 충남대학교 공과대학 공지사항 크롤링 시작 ===")
        print(f"크롤링 대상: 최대 {max_pages}페이지")
        print(f"대상 URL: {self.notice_url}")

        for page in range(1, max_pages + 1):
            print(f"\n[페이지 {page}/{max_pages}]")

            # 공지사항 목록 가져오기
            soup = self.get_notice_list(page)
            if not soup:
                print(f"  페이지 {page} 건너뜀")
                continue

            # 공지사항 링크 추출
            notices = self.extract_notice_links(soup)
            if not notices:
                print(f"  페이지 {page}에서 공지사항을 찾을 수 없음")
                continue

            print(f"  페이지 {page}에서 {len(notices)}개의 공지사항 발견")

            # 각 공지사항의 내용 가져오기
            for i, notice in enumerate(notices, 1):
                print(f"    [{i}/{len(notices)}] {notice['title'][:40]}...")

                # 내용 가져오기 (제목도 함께 전달)
                content = self.get_notice_content(notice['url'], notice['title'])

                notice_data = {
                    'page': page,
                    'index': i,
                    'title': notice['title'],
                    'date': notice['date'],
                    'url': notice['url'],
                    'content': content
                }

                all_notices.append(notice_data)

                # 서버 과부하 방지를 위한 딜레이
                time.sleep(2)

        return all_notices

    def save_to_json(self, notices, filename='notices.json'):
        """크롤링한 데이터를 JSON 파일로 저장"""
        try:
            # 요약 정보 추가
            summary = {
                'crawl_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_count': len(notices),
                'source_url': self.notice_url,
                'data': notices
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"\n=== 크롤링 완료 ===")
            print(f"저장 파일: {filename}")
            print(f"총 공지사항 수: {len(notices)}개")

            # 내용이 있는 공지사항과 없는 공지사항 통계
            with_content = sum(1 for notice in notices if notice['content'])
            without_content = len(notices) - with_content
            print(f"내용 추출 성공: {with_content}개")
            print(f"내용 추출 실패: {without_content}개")

            # 샘플 출력
            if notices:
                print(f"\n=== 샘플 데이터 ===")
                sample = notices[0]
                print(f"제목: {sample['title']}")
                print(f"날짜: {sample['date']}")
                print(f"URL: {sample['url']}")
                print(f"내용 미리보기: {sample['content'][:100]}...")

            return True

        except Exception as e:
            print(f"JSON 저장 오류: {e}")
            return False


def main():
    """메인 실행 함수"""
    print("충남대학교 공과대학 공지사항 크롤러 v2.1 (개선된 내용 추출)")
    print("=" * 60)

    crawler = CNUNoticeCrawler()

    try:
        # 3페이지까지 크롤링
        notices = crawler.crawl_notices(max_pages=10)

        if notices:
            # JSON 파일로 저장
            SAVE_DIR = Path("../rag_data/notice")
            out_path = SAVE_DIR / "notices.json"

            success = crawler.save_to_json(notices, filename=str(out_path))

            if success:
                print(f"\n✅ 크롤링이 성공적으로 완료되었습니다!")
                print(f"📁 파일 위치: ./notices.json")
            else:
                print(f"\n❌ 파일 저장에 실패했습니다.")
        else:
            print(f"\n⚠️  크롤링된 데이터가 없습니다.")
            print("사이트 구조가 변경되었거나 접근이 제한되었을 수 있습니다.")

    except KeyboardInterrupt:
        print(f"\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()