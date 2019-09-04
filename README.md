# 1. Github 시작하기_190629
* Github 가입 : https://github.com
* Github 필요 파일 설치 : https://gitforwindows.org (무조건 Next 누르면 됨)

### Github 상에 프로젝트 공간 생성(Online 내의 작업 공간)
1. Github 최우측 상단 프로필 Image 옆 '콤보박스' 클릭
2. Github Your respositories 클릭
3. 녹색 New 클릭
4. 만들고자 하는 프로젝트 이름을 Repository name에 입력 후 Creat Repository 클릭

### 내 컴퓨터 상에 프로젝트 공간 생성(Offline 내의 작업 공간)
1. 아무곳에나 새로운 폴더 만들기 - 프로젝트용
2. 아무런 텍스트 파일 만들기, 내용 아무렇게나 입력 - 파일 명 : number.txt 
3. 폴더 상에서 우측 마우스 클릭 후 Git Bash Here클릭 - 명령어 프롬포트 열림
4. git config --global user.name "your name" : "your name"에 github 내의 이름 적기 - git commit에 사용될 이름 
5. git config --global user.email "your_email@example.com" : "your_email@example.com"에 github 가입시 적은 email 적기 - git commit에 사용될 email
6. git config --list : 설정된 사항 확인 가능
7. git init : 새로운 git 저장소로 지정
8. git status : 현재 git 저장소 상황 확인 가능 - number.txt가 빨간색으로 되어있음 확인 가능
9. git add 파일명.확장자 : 온라인 공간으로 넘어가기 전의 index에 저장 - 다시 git status를 하면, 초록색으로 변한 것 확인 가능
10. git commit -m "first init" : index에 확정 저장 및 변경 내용에 대한 내용을 "내용"에 기입하여 진행
11. git remote add origin https://github repository 주소 : github 내 repository(프로젝트 공간)과 연결
12. git push origin master : 파일 올리기


